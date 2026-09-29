# Copyright (c) 2025, Infineon Technologies AG, or an affiliate of Infineon Technologies AG. All rights reserved.
# This software, associated documentation and materials ("Software") is owned by Infineon Technologies AG or one
# of its affiliates ("Infineon") and is protected by and subject to worldwide patent protection, worldwide copyright laws,
# and international treaty provisions. Therefore, you may use this Software only as provided in the license agreement accompanying
# the software package from which you obtained this Software. If no license agreement applies, then any use, reproduction, modification,
# translation, or compilation of this Software is prohibited without the express written permission of Infineon.
# Disclaimer: UNLESS OTHERWISE EXPRESSLY AGREED WITH INFINEON, THIS SOFTWARE IS PROVIDED AS-IS, WITH NO WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, ALL WARRANTIES OF NON-INFRINGEMENT OF THIRD-PARTY RIGHTS AND IMPLIED WARRANTIES
# SUCH AS WARRANTIES OF FITNESS FOR A SPECIFIC USE/PURPOSE OR MERCHANTABILITY. Infineon reserves the right to make changes to the Software
# without notice. You are responsible for properly designing, programming, and testing the functionality and safety of your intended application
# of the Software, as well as complying with any legal requirements related to its use. Infineon does not guarantee that the Software will be
# free from intrusion, data theft or loss, or other breaches ("Security Breaches"), and Infineon shall have no liability arising out of any
# Security Breaches. Unless otherwise explicitly approved by Infineon, the Software may not be used in any application where a failure of the
# Product or any consequences of the use thereof can reasonably be expected to result in personal injury.

from pathlib import Path

from .config import (
    ONNX2C_INCLUDE,
    TRICORE_LINKER_SCRIPT,
    TRICORE_CRT,
    ARM_LINKER_SCRIPT,
    ARM_CRT,
    TSIM_CMD,
    TSIM_MCONFIG,
    NSIM_TCF,
)

# Per-target TSIM memory configs live next to the linker scripts. TC3xx and
# TC4Dx differ in cache sizes and flash wait-states, so each points at its own
# MConfig (see AGENTS.md "TSIM MConfig" for the representative values used).
# Falls back to the generic TSIM_MCONFIG from .env if a per-target file is
# absent. TSIM_MCONFIG being set is still the "TSIM is available" signal.
_MCONFIG_DIR = Path(__file__).resolve().parent.parent / "Tools" / "linker_scripts"


def _tsim_mconfig(filename: str) -> str:
    """Resolve a per-target MConfig path, falling back to the generic one."""
    candidate = _MCONFIG_DIR / filename
    if candidate.exists():
        return str(candidate)
    # In the model-zoo deployment the package ships under _Tools/python, so the
    # package-relative Tools/linker_scripts above does not exist. The per-target
    # MConfigs are instead shipped next to the generic TSIM_MCONFIG.
    if TSIM_MCONFIG:
        alt = Path(TSIM_MCONFIG).resolve().parent / filename
        if alt.exists():
            return str(alt)
    return TSIM_MCONFIG


class TC4Dx:
    """TC4Dx TriCore CPU device configuration."""

    toolchain = "tricore"
    clock_freq_hz = 500_000_000  # 500 MHz

    # Pre-compile memory-fit check: map onnx2c report categories to the
    # MEMORY regions declared in the linker script. Constants (weights) land in
    # flash, activation variables in data RAM.
    memory_regions = {
        "constants": ("int_flash", TRICORE_LINKER_SCRIPT),
        "variables": ("int_dsprcpu0", TRICORE_LINKER_SCRIPT),
    }

    default_compiler_opts = [
        "-Ofast",
        "-mcpu=tc4DAx",
        # Use software double-precision. The inference kernel is single-precision
        # only, so this does not affect measured inference cycles; it only makes
        # the test harness' printf("%e", float) double conversion a library call
        # instead of the hardware FTODF opcode. TSIM's tc18 model does not
        # implement the TC1.8 double-precision FPU (FTODF -> TRAP 0x22 UOPC ->
        # infinite trap loop), so hardware double would hang the simulator.
        # (TC3/tc162 has no hardware double FPU at all, so it is always soft.)
        "-msoft-dp-float",
        "-save-temps",
        "-Wl,-gc-sections",
        "-Wl,--extmap=a",
        "-nocrt0",
        "-Xlinker",
        "--mcpu=tc18",
        f"-I{ONNX2C_INCLUDE}",
        f"-T{TRICORE_LINKER_SCRIPT}",
        TRICORE_CRT,
    ]

    default_emulator_opts = [
        "-display",
        "none",
        "-M",
        "tricore_tsim18",
        "-semihosting",
    ]

    default_onnx2c_opts = [
        "-punionize",
        "--comp_math",
        "-l3",
        "--mtc18",
        "--memconfig=22",
        "--comp_pattern=1",
        "--comp_direct",
    ]

    # Requires the newer tsim16p_e build (TC_MODELS_1.18.196) which adds the
    # TC1.8 model. The plain "-tc18" flag selects an empty/invalid model (blank
    # banner) and misbehaves; the concrete step variants must be used instead.
    # "-tc18_bstep" = "TC18 HV" (B-step, hypervisor) matches production TC4Dx.
    default_tsim_opts = (
        [
            "-MConfig",
            _tsim_mconfig("tsim_MConfig_tc4"),
            "-tc18_bstep",
            "-disable-watchdog",
        ]
        if TSIM_CMD and TSIM_MCONFIG
        else []
    )


class TC3xx:
    """TC3xx TriCore CPU device configuration."""

    toolchain = "tricore"
    clock_freq_hz = 300_000_000  # 300 MHz

    # Pre-compile memory-fit check (see TC4Dx.memory_regions).
    memory_regions = {
        "constants": ("int_flash", TRICORE_LINKER_SCRIPT),
        "variables": ("int_dsprcpu0", TRICORE_LINKER_SCRIPT),
    }

    default_compiler_opts = [
        "-Ofast",
        "-mcpu=tc39xx",
        "-save-temps",
        "-Wl,-gc-sections",
        "-Wl,--extmap=a",
        "-nocrt0",
        "-Xlinker",
        "--mcpu=tc162",
        f"-I{ONNX2C_INCLUDE}",
        f"-T{TRICORE_LINKER_SCRIPT}",
        TRICORE_CRT,
    ]

    default_emulator_opts = [
        "-display",
        "none",
        "-M",
        "tricore_tsim162",
        "-semihosting",
    ]

    default_onnx2c_opts = [
        "-punionize",
        "--comp_math",
        "-l3",
        "--mtc162",
        "--memconfig=22",
        "--comp_pattern=1",
        "--comp_direct",
    ]

    default_tsim_opts = (
        [
            "-MConfig",
            _tsim_mconfig("tsim_MConfig_tc3"),
            "-tc162p",
            "-disable-watchdog",
        ]
        if TSIM_CMD and TSIM_MCONFIG
        else []
    )


class TC4Dx_PPU:
    """TC4Dx PPU device configuration (LLVM/clang + nSIM)."""

    toolchain = "ppu"
    clock_freq_hz = 454_000_000  # 454 MHz

    default_compiler_opts = [
        "-O2",
        "-march=av2hs",
        "-mcpu=tc4d",
        "-w",
        "-lnsim",
        "-lcrt0",
        f"-I{ONNX2C_INCLUDE}",
    ]

    default_emulator_opts = ([f"--tcf={NSIM_TCF}"] if NSIM_TCF else []) + [
        "-prop=nsim_emt=1",
        "-prop=nsim_print_stats_on_exit=1",
    ]

    default_onnx2c_opts = [
        "-punionize",
        "--comp_math",
        "-l3",
        "--arcppu256",
        # TC4x PPU layout: 116 KB effective VCCM, 512 KB CSM_RW, and flash.
        # This is the ACS package's recommended device-realistic default.
        "--memconfig=101",
        "--comp_pattern=1",
        "--comp_direct",
    ]

    default_tsim_opts = []  # TSIM not applicable to PPU


class ARM_M4:
    """ARM Cortex-M4 device configuration (QEMU MPS2-AN386)."""

    toolchain = "arm"
    clock_freq_hz = 160_000_000  # 160 MHz

    # Pre-compile memory-fit check: constants (weights) -> FLASH, activation
    # variables -> RAM, sized from the MEMORY block in arm_mps2.ld.
    memory_regions = {
        "constants": ("FLASH", ARM_LINKER_SCRIPT),
        "variables": ("RAM", ARM_LINKER_SCRIPT),
    }

    default_compiler_opts = [
        "-mcpu=cortex-m4",
        "-mthumb",
        "-mfloat-abi=hard",
        "-mfpu=fpv4-sp-d16",
        "-Ofast",
        "--specs=rdimon.specs",
        "-nostartfiles",
        f"-I{ONNX2C_INCLUDE}",
        f"-T{ARM_LINKER_SCRIPT}" if ARM_LINKER_SCRIPT else None,
        ARM_CRT if ARM_CRT else None,
        "-lm",
    ]
    # Filter out None entries (missing config)
    default_compiler_opts = [o for o in default_compiler_opts if o is not None]

    default_emulator_opts = [
        "-machine",
        "mps2-an386",
        "-cpu",
        "cortex-m4",
        "-nographic",
        "-semihosting-config",
        "enable=on,target=native",
    ]

    default_onnx2c_opts = [
        "-punionize",
        "--comp_math",
        "-l3",
        "--armm4",
        "--memconfig=201",
        "--comp_pattern=1",
        "--comp_direct",
    ]

    default_tsim_opts = []  # TSIM not applicable to ARM


# Registry of available targets (name → device class)
TARGETS = {
    "tc4dx": TC4Dx,
    "tc3xx": TC3xx,
    "tc4dx_ppu": TC4Dx_PPU,
    "arm_m4": ARM_M4,
}
