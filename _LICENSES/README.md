# Third-Party Tool Licenses

The Docker image built for this model zoo bundles several third-party tools that
are used by the neural-network conversion and benchmarking pipeline. Each tool
is distributed under its own license. The full license texts are included in the
respective directories here and/or inside the tool distribution archives, and are
copied into the container at `/home/ubuntu/LICENSES`.

| Component | Used for | License location |
| --- | --- | --- |
| onnx2c (onnx2c-ifx 1.1.0) | ONNX → C code generation | [`LICENSES_ONNX2C/`](LICENSES_ONNX2C/); also ships its own `Linux/onnx2c-ifx/LICENSE` and `licenses/` in the IDC-downloaded ACS Edge AI package |
| AURIX&trade; GCC (TriCore&trade;) | TC3x / TC4x cross-compilation | [`LICENSES_GCC/`](LICENSES_GCC/) (GPL — see [`../source.txt`](../source.txt)); downloaded from IDC during setup |
| Arm&reg; GCC (`gcc-arm-none-eabi`, newlib) | Arm&reg; Cortex&reg;-M4/M7 cross-compilation | GPL / newlib license — installed from Ubuntu packages; see `/usr/share/doc/gcc-arm-none-eabi` and `/usr/share/doc/libnewlib-arm-none-eabi` in the container |
| QEMU (TriCore&trade; + Arm&reg;) | Instruction-set emulation | GPLv2 — bundled with the QEMU source build under `_Tools/qemu_build` |
| ARC LLVM / clang 21.1.8 | AURIX&trade; TC4x PPU cross-compilation | Apache 2.0 with LLVM exceptions — provided by the IDC-downloaded ACS Edge AI 1.0.0 package (`Linux/arc-llvm/LICENSE.TXT`, `EVALUATION_ONLY.TXT`) |
| nSIM 2025.12 (Synopsys ARC ISS) | PPU instruction-set simulation | Synopsys license — provided by the IDC-downloaded ACS Edge AI 1.0.0 package (`Linux/nSIM_64/LICENSE.txt`) |
| TSIM 1.18.196 (TriCore&trade; ISS) | Optional cycle-accurate TriCore&trade; simulation | Vendor license — downloaded as an IDC `.deb` during setup; the repository retains `_Tools/tsim/TSIM_Simulator_License.htm` |
| ACS Edge AI package 1.0.0 | Vendor bundle of onnx2c-ifx, ARC LLVM and nSIM | Vendor EULA — `Linux/license.txt` in the IDC-downloaded package (per-tool licenses listed above) |

> **Note:** The ARC LLVM, nSIM and TSIM tools are proprietary/third-party
> binaries downloaded from IDC according to `_CentralScripts/tool_loader/tools.csv`.
> Their licenses, as shipped inside the respective packages, govern their use.
> Please review them before use or redistribution. Downloaded packages are
> cached under `_CentralScripts/tool_loader/downloads/` and are not tracked.

For questions regarding licensing or to request specific source code versions,
contact AIModelZoo@infineon.com.
