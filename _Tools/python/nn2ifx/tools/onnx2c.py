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
import subprocess
import re
import numpy as np
import onnx
from onnx import numpy_helper
import logging
from ..config import ONNX2C_CMD
from ..emu_parsing import RUN_COMPLETION_MARKER

logger = logging.getLogger("nn2ifx.onnx2c")


class Onnx2cMemoryError(RuntimeError):
    """onnx2c aborted because the model does not fit the target's memory.

    Raised when onnx2c's internal constant/variable placement (e.g. the PPU
    ``--memconfig`` layout) cannot assign every tensor to a memory region. The
    ``summary`` attribute holds a concise, human-readable description of the
    unassigned tensor(s) and per-region usage extracted from onnx2c's output.
    """

    def __init__(self, summary: str):
        self.summary = summary
        super().__init__(summary)


def _parse_memory_abort(output: str) -> str | None:
    """Extract a concise memory-fit summary from onnx2c's output, or None.

    Returns None when the failure is not an out-of-memory abort, so the caller
    can fall back to the generic error path.
    """
    if "Memory constraints not fitting" not in output:
        return None

    unassigned = re.findall(r"Cannot fit to memory:.*?cname:\s*(\S+)", output)
    more = re.search(r"\((\d+) more tensor\(s\) unassigned\)", output)
    regions = re.findall(
        r"Memory\s+(\S+)\s+size\[bytes\]=(\d+)\s+used\[bytes\]=(\d+)", output
    )

    parts = []
    if unassigned:
        names = ", ".join(unassigned)
        parts.append(f"unassigned tensor(s): {names}")
    if more:
        parts.append(f"+{more.group(1)} more unassigned")
    for name, size, used in regions:
        pct = 100 * int(used) / int(size) if int(size) else 0
        parts.append(f"{name} {int(used):,}/{int(size):,} B ({pct:.0f}%)")

    if not parts:
        return "onnx2c reported the model does not fit the target's memory."
    return "; ".join(parts)


def generate(
    onnxpath: Path,
    cpath: Path,
    mainpath: Path,
    opts: list,
    test_data_dir: Path | None = None,
    profile: bool = False,
    log=None,
    toolchain: str = "tricore",
):
    """
    Generate C code from ONNX model using onnx2c.

    Args:
        onnxpath: Path to the input ONNX model file
        cpath: Path where the generated C code should be saved
        mainpath: Path where the main.c file should be saved
        opts: Options string to pass to onnx2c
        test_data_dir: Path to directory containing input_0.pb and output_0.pb
        profile: If True, add per-function instrumentation hooks to main.c
        log: Optional PipelineLog instance for structured logging
        toolchain: Target toolchain ("tricore", "arm", or "ppu")
    """
    # Ensure paths are Path objects
    onnxpath = Path(onnxpath)
    cpath = Path(cpath)
    mainpath = Path(mainpath)

    # Create parent directories if they don't exist
    cpath.parent.mkdir(parents=True, exist_ok=True)
    mainpath.parent.mkdir(parents=True, exist_ok=True)

    # Report path alongside the C file
    reportpath = cpath.with_suffix(".md")

    # Build the onnx2c command (report is always generated)
    cmd = [ONNX2C_CMD] + opts + [f"-r{reportpath}", str(onnxpath)]

    logger.info(f"Generating C code: {' '.join(cmd)}")
    if log:
        log.begin_step("ONNX to C Code Generation")
        log.log_command(cmd)

    # Run onnx2c and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"onnx2c stderr: {result.stderr}")
        combined = f"{result.stdout}\n{result.stderr}"
        mem_summary = _parse_memory_abort(combined)
        if mem_summary:
            logger.error(f"onnx2c: model does not fit target memory: {mem_summary}")
            if log:
                log.log_error("onnx2c: model does not fit the target's memory")
                log.log_error(mem_summary)
                log.log_output(result.stderr, "stderr")
            raise Onnx2cMemoryError(mem_summary)
        if log:
            log.log_error(f"onnx2c failed with return code {result.returncode}")
            log.log_output(result.stderr, "stderr")
        raise RuntimeError(f"onnx2c failed with return code {result.returncode}")

    # Log onnx2c debug output (from -l3)
    if log:
        log.log_output(result.stderr, "onnx2c log")

    # Write the generated C code to the output file
    with open(cpath, "w") as f:
        f.write(result.stdout)

    logger.info(f"Generated C code saved to: {cpath}")
    if reportpath.exists():
        logger.info(f"Generated report saved to: {reportpath}")

    if log:
        log.log_info(f"Generated C code: {cpath}")
        log.log_info(f"Generated report: {reportpath}")

    # Generate the main.c file
    generate_main(
        onnxpath, cpath, mainpath, test_data_dir, profile=profile, toolchain=toolchain
    )

    return cpath, mainpath


def generate_main(
    onnxpath: Path,
    cpath: Path,
    mainpath: Path,
    test_data_dir: Path | None = None,
    profile: bool = False,
    toolchain: str = "tricore",
):
    """
    Generate main.c with embedded test data for standalone execution.

    Args:
        onnxpath: Path to the ONNX model (to extract input/output shapes)
        cpath: Path to the generated C code (to find entry function)
        mainpath: Path where main.c should be saved
        test_data_dir: Path to directory containing input_0.pb and output_0.pb
        profile: If True, add per-function instrumentation hooks
        toolchain: Target toolchain ("tricore", "arm", or "ppu")
    """
    # Load ONNX model to get input/output shapes
    model = onnx.load(str(onnxpath))

    # Extract input shape
    input_shape = []
    for input_tensor in model.graph.input:
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        input_shape.append(shape)

    # Extract output shape
    output_shape = []
    for output_tensor in model.graph.output:
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        output_shape.append(shape)

    # Load test data if provided (need shapes for size calculation)
    if test_data_dir:
        test_data_dir = Path(test_data_dir)
        input_pb = test_data_dir / "input_0.pb"
        output_pb = test_data_dir / "output_0.pb"

        tensor = onnx.TensorProto()
        with open(input_pb, "rb") as f:
            tensor.ParseFromString(f.read())
        input_data = numpy_helper.to_array(tensor)

        tensor = onnx.TensorProto()
        with open(output_pb, "rb") as f:
            tensor.ParseFromString(f.read())
        expected_output = numpy_helper.to_array(tensor)
    else:
        # Use zeros as default test data
        input_data = np.zeros(input_shape[0], dtype=np.float32)
        expected_output = np.zeros(output_shape[0], dtype=np.float32)

    # Derive sizes from actual loaded data
    input_size = input_data.size
    output_size = expected_output.size
    logger.info(f"Input size: {input_size}, Output size: {output_size}")

    # Format C array dimensions e.g. [1][28][28][1]. A scalar (0-D) tensor has
    # no dimensions; emit a single-element array so the C declaration is valid.
    def shape_to_c_dims(shape):
        if len(shape) == 0:
            return "[1]"
        return "".join(f"[{max(d, 1)}]" for d in shape)

    # Format multi-dimensional C array initializer with nested braces
    def ndarray_to_c_initializer(arr):
        """Recursively format ndarray as nested C initializer."""
        if arr.ndim == 0:
            return "{" + f"{arr.item():.8e}f" + "}"
        if arr.ndim == 1:
            return "{" + ", ".join(f"{v:.8e}f" for v in arr) + "}"
        return (
            "{"
            + ", ".join(ndarray_to_c_initializer(arr[i]) for i in range(arr.shape[0]))
            + "}"
        )

    input_c_dims = shape_to_c_dims(input_data.shape)
    output_c_dims = shape_to_c_dims(expected_output.shape)
    input_c_init = ndarray_to_c_initializer(input_data)
    expected_c_init = ndarray_to_c_initializer(expected_output)

    # Generate the main.c template
    profile_hooks = ""
    if profile:
        if toolchain == "arm":
            profile_hooks = """
/* Per-function instrumentation via CPI plugin address markers.
 * Since QEMU doesn't emulate DWT CYCCNT for ARM MPS2, we write to fixed
 * RAM addresses that the CPI plugin monitors to track instruction counts.
 * These addresses (0x20200000/04) are in valid SRAM, reserved for profiling.
 */
#define PROFILE_MARKER_ENTER (*(volatile unsigned int *)0x20200000)
#define PROFILE_MARKER_EXIT  (*(volatile unsigned int *)0x20200004)

__attribute__((no_instrument_function))
void __cyg_profile_func_enter(void *func, void *caller) {
    PROFILE_MARKER_ENTER = (unsigned int)func;
}

__attribute__((no_instrument_function))
void __cyg_profile_func_exit(void *func, void *caller) {
    PROFILE_MARKER_EXIT = (unsigned int)func;
    printf("PROFILE: 0x%08x\\n", (unsigned int)func);
}
"""
        elif toolchain == "tricore":
            profile_hooks = """
/* Per-function instrumentation hooks (used with -finstrument-functions) */
#define MAX_PROFILE_DEPTH 64
static unsigned int _profile_stack[MAX_PROFILE_DEPTH];
static int _profile_depth = 0;

__attribute__((no_instrument_function))
void __cyg_profile_func_enter(void *func, void *caller) {
    unsigned int icnt;
    __asm volatile ("mfcr %0, 0xfc08" : "=d" (icnt));
    if (_profile_depth < MAX_PROFILE_DEPTH) {
        _profile_stack[_profile_depth] = icnt;
    }
    _profile_depth++;
}

__attribute__((no_instrument_function))
void __cyg_profile_func_exit(void *func, void *caller) {
    _profile_depth--;
    unsigned int icnt;
    __asm volatile ("mfcr %0, 0xfc08" : "=d" (icnt));
    if (_profile_depth >= 0 && _profile_depth < MAX_PROFILE_DEPTH) {
        unsigned int delta = icnt - _profile_stack[_profile_depth];
        printf("PROFILE: 0x%08x %u\\n", (unsigned int)func, delta);
    }
}
"""

    # Instruction counter code depends on toolchain
    if toolchain == "arm":
        icnt_code = """
/* DWT Cycle Counter for ARM Cortex-M (enable DWT->CYCCNT) */
__attribute__((no_instrument_function))
static inline void enable_cycle_counter(void) {
    volatile unsigned int *DWT_CTRL = (volatile unsigned int *)0xE0001000;
    volatile unsigned int *DWT_CYCCNT = (volatile unsigned int *)0xE0001004;
    volatile unsigned int *CoreDebug_DEMCR = (volatile unsigned int *)0xE000EDFC;
    *CoreDebug_DEMCR |= (1 << 24);  /* TRCENA */
    *DWT_CTRL |= 1;                  /* CYCCNTENA */
    *DWT_CYCCNT = 0;                 /* Reset counter */
}

__attribute__((no_instrument_function))
static inline unsigned int read_icnt(void) {
    return *((volatile unsigned int *)0xE0001004);  /* DWT->CYCCNT */
}
"""
    elif toolchain == "ppu":
        icnt_code = """
/* PPU: no hardware instruction counter available in nSIM.
 * Runtime estimation will use nSIM's reported cycle count instead. */
static inline unsigned int read_icnt(void) {
    return 0;
}
"""
    else:
        icnt_code = """
/* CPU Instruction Counter access for TriCore (QEMU emulates ICNT at 0xFC08) */
__attribute__((no_instrument_function))
static inline unsigned int read_icnt(void) {
    unsigned int ret;
    __asm volatile ("mfcr %0, 0xfc08" : "=d" (ret));
    return ret;
}
"""

    # ARM needs to call enable_cycle_counter() before use
    icnt_init = "    enable_cycle_counter();\n" if toolchain == "arm" else ""

    # PPU uses __vccm address space for output buffers
    if toolchain == "ppu":
        entry_decl = "void entry(const float *input, float __vccm *output);"
        # entry() writes the result via vector stores into VCCM, so the output
        # buffer must be allocated in VCCM (address space 4) rather than RAM.
        output_storage = f"static float __vccm actual_output{output_c_dims};"
        vccm_define = "#define __vccm __attribute__((address_space(4)))"
        # Read results back through a __vccm pointer so loads come from VCCM.
        actual_ptr_decl = "float __vccm *actual = (float __vccm *)actual_output;"
    else:
        entry_decl = "void entry(const float *input, float *output);"
        output_storage = f"static float actual_output{output_c_dims};"
        vccm_define = ""
        actual_ptr_decl = "float *actual = (float *)actual_output;"

    # PPU entry() takes a __vccm pointer into the VCCM output buffer
    if toolchain == "ppu":
        entry_call = "entry((const float *)input_data, (float __vccm *)actual_output);"
    else:
        entry_call = "entry((const float *)input_data, (float *)actual_output);"

    # ARM has no guest-readable instruction counter (QEMU MPS2 doesn't emulate
    # DWT CYCCNT). To still measure the *inference-only* region — matching the
    # TriCore ICNT isolation — bracket entry() with writes to the CPI plugin's
    # marker addresses. The plugin emits "PROFILE_INSN: <delta>" for the region.
    # Skip in --profile mode, where -finstrument-functions already brackets
    # every function (including entry()) via the same markers.
    if toolchain == "arm" and not profile:
        infer_marker_pre = (
            "*(volatile unsigned int *)0x20200000 = 1u; /* inference begin */\n    "
        )
        infer_marker_post = (
            "\n    *(volatile unsigned int *)0x20200004 = 1u; /* inference end */"
        )
    else:
        infer_marker_pre = ""
        infer_marker_post = ""

    main_c_code = f"""#include <stdio.h>
#include <math.h>
{vccm_define}
{profile_hooks}
{icnt_code}

{entry_decl}

static float input_data{input_c_dims} = {input_c_init};

static float expected_output{output_c_dims} = {expected_c_init};

{output_storage}

__attribute__((no_instrument_function))
int main() {{
{icnt_init}    // Measure inference instruction count
    unsigned int icnt_start = read_icnt();
    {infer_marker_pre}{entry_call}{infer_marker_post}
    unsigned int icnt_end = read_icnt();
    unsigned int icnt_diff = icnt_end - icnt_start;

    // Print results and compute error
    float max_abs_err = 0.0f;
    float sum_sq_err = 0.0f;
    {actual_ptr_decl}
    float *expected = (float *)expected_output;

    printf("=== Inference Results ===\\n");
    for (int i = 0; i < {output_size}; i++) {{
        float err = actual[i] - expected[i];
        float abs_err = (err < 0) ? -err : err;
        if (abs_err > max_abs_err) max_abs_err = abs_err;
        sum_sq_err += err * err;
        printf("out[%d]: actual=%e  expected=%e  err=%e\\n",
               i, actual[i], expected[i], err);
    }}

    float rmse = sqrtf(sum_sq_err / {output_size});
    printf("\\n=== Error Summary ===\\n");
    printf("Max absolute error: %e\\n", max_abs_err);
    printf("RMSE: %e\\n", rmse);
    printf("\\n=== Performance ===\\n");
    if (icnt_diff > 0)
        printf("Inference instructions: %u\\n", icnt_diff);
    else
        printf("Inference instructions: N/A (no HW counter)\\n");

    printf("{RUN_COMPLETION_MARKER}\\n");
    return 0;
}}
"""

    # Write the main.c file
    with open(mainpath, "w") as f:
        f.write(main_c_code)

    logger.info(f"Generated main.c saved to: {mainpath}")

    return mainpath
