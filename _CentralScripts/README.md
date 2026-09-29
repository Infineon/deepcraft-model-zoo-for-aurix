# Central Scripts

This folder contains scripts for common functions needed across various AI model projects in the model zoo, including:

For repository setup and an overview of the model examples, see the [repository README](../README.md).

- **Helper Functions**: Shared utilities for model conversion, validation, and analysis
- **Requirements Management**: Centralized dependency specifications and validation
- **Testing Utilities**: Common test functions and validation scripts

## Key Components

- **`helper_functions.py`**: Main utility library containing functions for:
  - ONNX model conversion and export (TensorFlow ↔ PyTorch ↔ ONNX)
  - Data preprocessing and tensor operations
  - Docker container management for model compilation

- **`requirements.txt`**: Centralized Python dependencies for the entire model zoo ecosystem

- **`test_requirements.py`**: pytest checks for:
  - Package installation and compatibility
  - Critical package imports
  - Version compatibility checks
  - Jupyter, data-science, and image/audio package availability

- **`tool_loader/`**: Secure IDC downloader used by `setup.sh` to obtain the
  manifest-pinned ACS Edge AI, AURIX GCC, and TSIM packages. Repository setup is
  the supported entry point; see [the tool-loader documentation](tool_loader/README.md)
  for direct troubleshooting commands.

## Conversion service (REST API)

The Docker image runs a Flask service on port `8080` that converts an ONNX model
to C, compiles it, and benchmarks it on the selected hardware target(s). The
`CallTools` class in `python_flask_client.py` is the convenience wrapper used by
the notebooks.

### Supported targets

| Target | Hardware |
| --- | --- |
| `tc3xx` | AURIX&trade; TriCore&trade; TC3x |
| `tc4dx` | AURIX&trade; TriCore&trade; TC4x |
| `tc4dx_ppu` | AURIX&trade; TC4x Parallel Processing Unit (PPU) |
| `arm_m4` | Arm&reg; Cortex&reg;-M4 |

Legacy aliases `TC3` / `TC4` are still accepted. The keyword `compare` (or
`all`) benchmarks the model on **every** target in one call.

For the `tc4dx_ppu` target the reported cycle count is *estimated* (nSIM is
functional-only). See [ppu_cpi_model.md](ppu_cpi_model.md) for how the PPU cycle
model works, its memory-hierarchy corrections, and its accuracy vs. hardware.

### `POST /convert`

Multipart form fields:

| Field | Required | Description |
| --- | --- | --- |
| `onnx-file` | yes | the ONNX model |
| `input_0` | yes | reference input tensor (`.pb`) |
| `output_0` | yes | reference output tensor (`.pb`) |
| `target` | no | one target, a space/comma separated list, or `compare`/`all`; defaults to `tc3xx` |
| `tsim` | no | TriCore timing backend. **TSIM is the default** — omit the field or send a truthy value for TSIM, send a falsy value (`false`) to use the QEMU+CPI alternative. Ignored for ARM/PPU. |
| `profile` | no | `true` to enable per-node instruction profiling |

The response is a JSON envelope containing `status` and an *artifact name →
download URL* map. Each artifact is fetched from `GET /download/<path>`.

The service returns an envelope with `status` and `artifacts` fields. A complete
conversion has status `ok`; incomplete comparisons use `partial` and failed
comparisons use `no_eligible_results`. These responses use HTTP 422. Uploads
are limited by `ZOO_MAX_CONTENT_LENGTH` (512 MiB by default), and the service
rejects requests missing any required file. The default helper publishes the
container only on localhost.

- **Single target** returns: `model.c` (the generated neural-network C code, for
  integration into your own project), `main.c`, `model.elf`, `model.md`
  (onnx2c-ifx report), `results.json` (harmonized metrics), `pipeline.log`, and
  `model.tsim_prof.log` (present on TriCore TSIM runs, i.e. the default).
- **Compare** additionally returns `comparison.txt` / `comparison.json` plus the
  per-target artifacts under keys like `tc4dx/model.c`.

Unavailable targets are kept in the comparison with a short reason and, for
pipeline failures, a relative reference such as `arm_m4/pipeline.log`. Detailed
memory placement, compiler, simulator, and emulator diagnostics remain in the
target's `pipeline.log` and `results.json`; they are intentionally omitted from
the aggregate comparison and notebook table.

### Using `CallTools`

Per-target artifacts are always saved into a hardware-specific subfolder, e.g.
`<model_folder>/tc4dx/model.c`.

```python
from _CentralScripts.python_flask_client import CallTools

# (a) One platform at a time — produces <model_folder>/<target>/ for each.
for target in ["tc3xx", "tc4dx", "arm_m4", "tc4dx_ppu"]:
    CallTools(folder=model_folder, target=target).convert_model()

# (b) Compare across all platforms in a single call — produces the same
#     per-target subfolders PLUS an aggregated comparison.txt/comparison.json
#     in <model_folder>/.
CallTools(folder=model_folder, target="compare").convert_model()
```

> **Which to use?** The plain loop in (a) runs each target independently and does
> **not** generate a comparison table. To obtain the side-by-side
> `comparison.txt` / `comparison.json`, use the compare mode in (b) (or pass an
> explicit list, e.g. `target="tc3xx tc4dx arm_m4"`). Both approaches write the
> same `<model_folder>/<target>/` subfolders, so the helper visualizations work
> either way.

Optional flags: `CallTools(folder=..., target="compare", profile=True)` adds
per-node profiling. TriCore targets are timed with TSIM by default; pass
`tsim=False` to use the QEMU+CPI alternative instead.


## Dependencies

Core dependencies include:

- Deep learning frameworks (TensorFlow, PyTorch)
- Model conversion tools (ONNX, tf2onnx)
- Data science stack (NumPy, Pandas, Matplotlib, Seaborn)
- Development tools (Jupyter, Flask)
- Testing tools (pytest)

See `requirements.txt` for the complete list with version specifications.

To verify the installed Python environment manually, run:

```bash
cd _CentralScripts
python -m pytest test_requirements.py -q
```

## License

Please see our [LICENSE](../LICENSE) for copyright and license information.
