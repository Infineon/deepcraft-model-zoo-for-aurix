/*
 * Copyright (c) 2025, Infineon Technologies AG, or an affiliate of Infineon Technologies AG. All rights reserved.
 * This software, associated documentation and materials ("Software") is owned by Infineon Technologies AG or one
 * of its affiliates ("Infineon") and is protected by and subject to worldwide patent protection, worldwide copyright laws,
 * and international treaty provisions. Therefore, you may use this Software only as provided in the license agreement accompanying
 * the software package from which you obtained this Software. If no license agreement applies, then any use, reproduction, modification,
 * translation, or compilation of this Software is prohibited without the express written permission of Infineon.
 * Disclaimer: UNLESS OTHERWISE EXPRESSLY AGREED WITH INFINEON, THIS SOFTWARE IS PROVIDED AS-IS, WITH NO WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, ALL WARRANTIES OF NON-INFRINGEMENT OF THIRD-PARTY RIGHTS AND IMPLIED WARRANTIES
 * SUCH AS WARRANTIES OF FITNESS FOR A SPECIFIC USE/PURPOSE OR MERCHANTABILITY. Infineon reserves the right to make changes to the Software
 * without notice. You are responsible for properly designing, programming, and testing the functionality and safety of your intended application
 * of the Software, as well as complying with any legal requirements related to its use. Infineon does not guarantee that the Software will be
 * free from intrusion, data theft or loss, or other breaches ("Security Breaches"), and Infineon shall have no liability arising out of any
 * Security Breaches. Unless otherwise explicitly approved by Infineon, the Software may not be used in any application where a failure of the
 * Product or any consequences of the use thereof can reasonably be expected to result in personal injury.
*/

/*
 * QEMU Plugin: CPI-weighted Instruction Counter
 *
 * Supports TriCore and ARM Cortex-M targets.
 * Counts executed instructions weighted by estimated CPI (Cycles Per Instruction)
 * to provide a more realistic runtime estimation than flat instruction counts.
 *
 * Build:
 *   TriCore: gcc -shared -fPIC -o libcpi_counter.so cpi_counter.c -I. (uses qemu-plugin.h, API v1)
 *   ARM:     gcc -shared -fPIC -DTARGET_ARM -o libcpi_counter_arm.so cpi_counter.c -I. -include qemu-plugin-arm.h
 *
 * Usage:
 *   qemu-system-tricore -plugin ./libcpi_counter.so -kernel model.elf ...
 *   qemu-system-arm     -plugin ./libcpi_counter_arm.so -kernel model.elf ...
 *
 * Output (printed to stderr at exit):
 *   CPI_RESULT: <total_cycles> <total_insn> <avg_cpi>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#ifdef TARGET_ARM
#include "qemu-plugin-arm.h"
#else
#include "qemu-plugin.h"
#endif

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

/* Cycle cost categories */
enum insn_class {
    INSN_ALU = 0,       /* Simple ALU: add, sub, mov, logic */
    INSN_LOAD,          /* Load from memory */
    INSN_STORE,         /* Store to memory */
    INSN_BRANCH,        /* Branch/call/jump */
    INSN_MUL,           /* Integer multiply */
    INSN_FPU_SIMPLE,    /* FPU add/sub/mul */
    INSN_FPU_MAC,       /* FPU madd/msub */
    INSN_FPU_DIV,       /* FPU division */
    INSN_FPU_SQRT,      /* FPU square root */
    INSN_CLASS_COUNT
};

/*
 * Architecture-specific CPI tables.
 *
 * TriCore TC1.6.2/TC1.8: Single-issue in-order, 4-5 stage pipeline.
 * Timings from Infineon TC1.6.2/TC1.8 Instruction Set Manual (timing tables).
 * Assumes DSPR/PSPR access (tightly coupled RAM, no wait states).
 * Same latencies apply to both TC3xx (TC1.6.2) and TC4Dx (TC1.8) — the
 * difference is TC1.8's improved throughput (back-to-back FPU MACs), which
 * a per-instruction latency model cannot capture.
 *
 * ARM Cortex-M4: Single-issue in-order.
 * Timings from ARM Cortex-M4 TRM (DDI0439).
 */
#ifdef TARGET_ARM
static const uint64_t cpi_table[INSN_CLASS_COUNT] = {
    [INSN_ALU]        = 1,   /* add, mov, logic — 1 cycle */
    [INSN_LOAD]       = 2,   /* ldr — 2 cycles (1 access + 1 result latency) */
    [INSN_STORE]      = 1,   /* str — 1 cycle (write buffer absorbs) */
    [INSN_BRANCH]     = 2,   /* b/bl — 1-3 cycles, avg 2 (pipeline refill) */
    [INSN_MUL]        = 1,   /* mul — 1 cycle (single-cycle multiplier) */
    [INSN_FPU_SIMPLE] = 1,   /* vadd/vsub/vmul.f32 — 1 cycle */
    [INSN_FPU_MAC]    = 3,   /* vfma/vmla.f32 — 3 cycle latency */
    [INSN_FPU_DIV]    = 14,  /* vdiv.f32 — 14 cycles */
    [INSN_FPU_SQRT]   = 14,  /* vsqrt.f32 — 14 cycles */
};
#else
static const uint64_t cpi_table[INSN_CLASS_COUNT] = {
    [INSN_ALU]        = 1,   /* add, mov, logic, shift — 1 cycle */
    [INSN_LOAD]       = 1,   /* ld.w from DSPR — 1 cycle */
    [INSN_STORE]      = 1,   /* st.w to DSPR — 1 cycle */
    [INSN_BRANCH]     = 2,   /* j/call/ret — 2 cycles (pipeline flush) */
    [INSN_MUL]        = 2,   /* mul 32×32 — 2 cycle latency */
    [INSN_FPU_SIMPLE] = 2,   /* add.f/sub.f/mul.f — 2 cycle latency */
    [INSN_FPU_MAC]    = 3,   /* madd.f/msub.f — 3 cycle latency */
    [INSN_FPU_DIV]    = 20,  /* div.f — ~18-22 cycles (data dependent) */
    [INSN_FPU_SQRT]   = 20,  /* sqrt.f — ~18-22 cycles */
};
#endif

/* Counters */
static uint64_t class_count[INSN_CLASS_COUNT];
static uint64_t total_insn;
static uint64_t total_cycles;

#ifdef TARGET_ARM
/*
 * Profiling marker support for ARM.
 *
 * Since QEMU doesn't emulate DWT CYCCNT for MPS2, the guest can't read an
 * instruction counter directly. Instead, the guest writes to fixed addresses
 * in the upper RAM region (below the stack) that this plugin monitors:
 *
 *   Store to 0x20200000 = "function enter" (plugin pushes insn count)
 *   Store to 0x20200004 = "function exit"  (plugin pops, computes delta)
 *
 * These addresses are in valid SRAM at the midpoint of the 4MB RAM region,
 * well away from both BSS (near start) and stack (at end).
 */
#define PROFILE_ADDR_ENTER 0x20200000ULL
#define PROFILE_ADDR_EXIT  0x20200004ULL

#define MAX_PROFILE_STACK 64
static uint64_t profile_insn_stack[MAX_PROFILE_STACK];
static uint64_t profile_cycle_stack[MAX_PROFILE_STACK];
static int profile_depth = 0;

static void mem_write_cb(unsigned int vcpu_idx, qemu_plugin_meminfo_t info,
                         uint64_t vaddr, void *userdata)
{
    if (vaddr == PROFILE_ADDR_ENTER) {
        if (profile_depth < MAX_PROFILE_STACK) {
            profile_insn_stack[profile_depth] = total_insn;
            profile_cycle_stack[profile_depth] = total_cycles;
        }
        profile_depth++;
    } else if (vaddr == PROFILE_ADDR_EXIT) {
        profile_depth--;
        if (profile_depth >= 0 && profile_depth < MAX_PROFILE_STACK) {
            uint64_t insn_delta = total_insn - profile_insn_stack[profile_depth];
            uint64_t cycle_delta = total_cycles - profile_cycle_stack[profile_depth];
            fprintf(stderr, "PROFILE_INSN: %" PRIu64 "\n", insn_delta);
            fprintf(stderr, "PROFILE_RESULT: %" PRIu64 " %" PRIu64 "\n",
                    insn_delta, cycle_delta);
        }
    }
}
#endif /* TARGET_ARM */

/*
 * Classify a TriCore instruction from its disassembly string.
 * QEMU format: "  <hex>    <mnemonic> <operands>"
 * Example: "  0002001d    j " or "  0147f08f    or %d0,%d0,127"
 */
static enum insn_class classify_tricore(const char *disas)
{
    if (!disas || !disas[0]) return INSN_ALU;

    /* Skip leading whitespace */
    while (*disas == ' ' || *disas == '\t') disas++;

    /* Skip hex bytes (alphanumeric) */
    while ((*disas >= '0' && *disas <= '9') ||
           (*disas >= 'a' && *disas <= 'f') ||
           (*disas >= 'A' && *disas <= 'F')) disas++;

    /* Skip whitespace between hex and mnemonic */
    while (*disas == ' ' || *disas == '\t') disas++;

    /* Now disas points to the mnemonic */
    if (!*disas) return INSN_ALU;

    /* FPU division */
    if (strncmp(disas, "div.f", 5) == 0) return INSN_FPU_DIV;

    /* FPU square root */
    if (strncmp(disas, "sqrt.f", 6) == 0) return INSN_FPU_SQRT;

    /* FPU MAC: madd.f, msub.f */
    if (strncmp(disas, "madd.f", 6) == 0 ||
        strncmp(disas, "msub.f", 6) == 0) return INSN_FPU_MAC;

    /* FPU simple: add.f, sub.f, mul.f, cmp.f, etc. */
    if (strncmp(disas, "add.f", 5) == 0 ||
        strncmp(disas, "sub.f", 5) == 0 ||
        strncmp(disas, "mul.f", 5) == 0 ||
        strncmp(disas, "cmp.f", 5) == 0 ||
        strncmp(disas, "ftoi", 4) == 0 ||
        strncmp(disas, "itof", 4) == 0 ||
        strncmp(disas, "ftoiz", 5) == 0 ||
        strncmp(disas, "ftouz", 5) == 0) return INSN_FPU_SIMPLE;

    /* Integer multiply: mul, madd, msub (without .f suffix) */
    if ((strncmp(disas, "mul", 3) == 0 && strncmp(disas, "mul.f", 5) != 0) ||
        (strncmp(disas, "madd", 4) == 0 && strncmp(disas, "madd.f", 6) != 0) ||
        (strncmp(disas, "msub", 4) == 0 && strncmp(disas, "msub.f", 6) != 0))
        return INSN_MUL;

    /* Loads: ld.* */
    if (strncmp(disas, "ld.", 3) == 0) return INSN_LOAD;

    /* Stores: st.* */
    if (strncmp(disas, "st.", 3) == 0) return INSN_STORE;

    /* Branches/jumps/calls */
    if (disas[0] == 'j' || disas[0] == 'J' ||  /* j, jl, ji, jeq, jne, ... */
        strncmp(disas, "call", 4) == 0 ||
        strncmp(disas, "ret", 3) == 0 ||
        strncmp(disas, "loop", 4) == 0 ||
        strncmp(disas, "rfe", 3) == 0) return INSN_BRANCH;

    /* Everything else: ALU */
    return INSN_ALU;
}

/*
 * Classify an ARM/Thumb instruction from its disassembly string.
 * QEMU ARM format: "<mnemonic>\t<operands>" or "<mnemonic><cond>\t<operands>"
 * Examples: "ldr\tr0, [sp, #4]", "vadd.f32\ts0, s1, s2", "bl\t#0x1234"
 */
static enum insn_class classify_arm(const char *disas)
{
    if (!disas || !disas[0]) return INSN_ALU;

    /* Skip leading whitespace (shouldn't be any, but be safe) */
    while (*disas == ' ' || *disas == '\t') disas++;

    if (!*disas) return INSN_ALU;

    /* FPU division: vdiv */
    if (strncmp(disas, "vdiv", 4) == 0) return INSN_FPU_DIV;

    /* FPU square root: vsqrt */
    if (strncmp(disas, "vsqrt", 5) == 0) return INSN_FPU_SQRT;

    /* FPU MAC: vmla, vmls, vfma, vfms, vnmla, vnmls */
    if (strncmp(disas, "vmla", 4) == 0 ||
        strncmp(disas, "vmls", 4) == 0 ||
        strncmp(disas, "vfma", 4) == 0 ||
        strncmp(disas, "vfms", 4) == 0 ||
        strncmp(disas, "vnmla", 5) == 0 ||
        strncmp(disas, "vnmls", 5) == 0) return INSN_FPU_MAC;

    /* FPU simple: vadd, vsub, vmul, vcmp, vcvt, vabs, vneg */
    if (strncmp(disas, "vadd", 4) == 0 ||
        strncmp(disas, "vsub", 4) == 0 ||
        strncmp(disas, "vmul", 4) == 0 ||
        strncmp(disas, "vcmp", 4) == 0 ||
        strncmp(disas, "vcvt", 4) == 0 ||
        strncmp(disas, "vabs", 4) == 0 ||
        strncmp(disas, "vneg", 4) == 0 ||
        strncmp(disas, "vmov", 4) == 0) return INSN_FPU_SIMPLE;

    /* Integer multiply: mul, mla, mls, umull, smull, smmul, smlal */
    if (strncmp(disas, "mul", 3) == 0 ||
        strncmp(disas, "mla", 3) == 0 ||
        strncmp(disas, "mls", 3) == 0 ||
        strncmp(disas, "umull", 5) == 0 ||
        strncmp(disas, "smull", 5) == 0 ||
        strncmp(disas, "smmul", 5) == 0 ||
        strncmp(disas, "smlal", 5) == 0) return INSN_MUL;

    /* FPU loads/stores: vldr, vstr */
    if (strncmp(disas, "vldr", 4) == 0) return INSN_LOAD;
    if (strncmp(disas, "vstr", 4) == 0) return INSN_STORE;

    /* Loads: ldr, ldm, ldrb, ldrh, ldrd, pop */
    if (strncmp(disas, "ldr", 3) == 0 ||
        strncmp(disas, "ldm", 3) == 0 ||
        strncmp(disas, "pop", 3) == 0) return INSN_LOAD;

    /* Stores: str, stm, strb, strh, strd, push */
    if (strncmp(disas, "str", 3) == 0 ||
        strncmp(disas, "stm", 3) == 0 ||
        strncmp(disas, "push", 4) == 0) return INSN_STORE;

    /* Branches: b, bl, bx, blx, cbz, cbnz, tbb, tbh */
    if ((disas[0] == 'b' && (disas[1] == '\t' || disas[1] == ' ' ||
         disas[1] == 'l' || disas[1] == 'x' || disas[1] == '.' ||
         disas[1] == 'e' || disas[1] == 'n' || disas[1] == 'c' ||
         disas[1] == 'h' || disas[1] == 'g' || disas[1] == 'm' ||
         disas[1] == 'p' || disas[1] == 'v')) ||
        strncmp(disas, "cbz", 3) == 0 ||
        strncmp(disas, "cbnz", 4) == 0 ||
        strncmp(disas, "tbb", 3) == 0 ||
        strncmp(disas, "tbh", 3) == 0) return INSN_BRANCH;

    /* Everything else: ALU (add, sub, mov, and, orr, eor, cmp, ...) */
    return INSN_ALU;
}

/*
 * Dispatch to architecture-specific classifier.
 */
static enum insn_class classify_from_disas(const char *disas)
{
#ifdef TARGET_ARM
    return classify_arm(disas);
#else
    return classify_tricore(disas);
#endif
}

/* Per-instruction execution callback */
static void insn_exec_cb(unsigned int vcpu_idx, void *userdata)
{
    enum insn_class cls = (enum insn_class)(uintptr_t)userdata;
    class_count[cls]++;
    total_insn++;
    total_cycles += cpi_table[cls];
}

/* Translation block callback — instrument each instruction */
static void tb_trans_cb(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    size_t n_insns = qemu_plugin_tb_n_insns(tb);

    for (size_t i = 0; i < n_insns; i++) {
        struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
        char *disas = qemu_plugin_insn_disas(insn);

        enum insn_class cls = classify_from_disas(disas);

        qemu_plugin_register_vcpu_insn_exec_cb(
            insn, insn_exec_cb,
            QEMU_PLUGIN_CB_NO_REGS,
            (void *)(uintptr_t)cls
        );

#ifdef TARGET_ARM
        /* Register memory write callback to detect profiling markers */
        qemu_plugin_register_vcpu_mem_cb(
            insn, mem_write_cb,
            QEMU_PLUGIN_CB_NO_REGS,
            QEMU_PLUGIN_MEM_W,
            NULL
        );
#endif

        if (disas) free(disas);
    }
}

/* Print results at exit */
static void plugin_exit(qemu_plugin_id_t id, void *userdata)
{
    static const char *class_names[INSN_CLASS_COUNT] = {
        "ALU", "Load", "Store", "Branch",
        "Mul", "FPU_simple", "FPU_MAC", "FPU_DIV", "FPU_SQRT"
    };

    fprintf(stderr, "\n=== CPI Plugin Results ===\n");
    fprintf(stderr, "%-12s %10s %6s %10s\n", "Class", "Count", "CPI", "Cycles");
    fprintf(stderr, "%-12s %10s %6s %10s\n", "----", "-----", "---", "------");

    for (int i = 0; i < INSN_CLASS_COUNT; i++) {
        if (class_count[i] > 0) {
            fprintf(stderr, "%-12s %10" PRIu64 " %6" PRIu64 " %10" PRIu64 "\n",
                    class_names[i], class_count[i],
                    cpi_table[i], class_count[i] * cpi_table[i]);
        }
    }

    double avg_cpi = total_insn > 0 ? (double)total_cycles / total_insn : 0.0;
    fprintf(stderr, "%-12s %10s %6s %10s\n", "----", "-----", "---", "------");
    fprintf(stderr, "%-12s %10" PRIu64 " %6.2f %10" PRIu64 "\n",
            "TOTAL", total_insn, avg_cpi, total_cycles);

    /* Machine-parseable line for Python */
    fprintf(stderr, "CPI_RESULT: %" PRIu64 " %" PRIu64 " %.4f\n",
            total_cycles, total_insn, avg_cpi);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id,
                                           const qemu_info_t *info,
                                           int argc, char **argv)
{
    /* Register translation callback */
    qemu_plugin_register_vcpu_tb_trans_cb(id, tb_trans_cb);

    /* Register exit callback */
    qemu_plugin_register_atexit_cb(id, plugin_exit, NULL);

    return 0;
}
