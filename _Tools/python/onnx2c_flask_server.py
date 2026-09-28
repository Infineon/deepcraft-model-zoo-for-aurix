# Copyright (c) 2025, Infineon Technologies AG, or an affiliate of Infineon Technologies AG. All rights reserved.
#
# This software, associated documentation and materials ("Software") is owned by Infineon Technologies AG or one of
# its affiliates ("Infineon") and is protected by and subject to worldwide patent protection, worldwide copyright laws,
# and international treaty provisions. Therefore, you may use this Software only as provided in the license agreement
# accompanying the software package from which you obtained this Software. If no license agreement applies, then any use,
# reproduction, modification, translation, or compilation of this Software is prohibited without the express written
# permission of Infineon.
#
# Disclaimer: UNLESS OTHERWISE EXPRESSLY AGREED WITH INFINEON, THIS SOFTWARE IS PROVIDED AS-IS, WITH NO WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, ALL WARRANTIES OF NON-INFRINGEMENT OF THIRD-PARTY RIGHTS AND IMPLIED
# WARRANTIES SUCH AS WARRANTIES OF FITNESS FOR A SPECIFIC USE/PURPOSE OR MERCHANTABILITY. Infineon reserves the right to make
# changes to the Software without notice. You are responsible for properly designing, programming, and testing the
# functionality and safety of your intended application of the Software, as well as complying with any legal requirements
# related to its use. Infineon does not guarantee that the Software will be free from intrusion, data theft or loss, or other
# breaches ("Security Breaches"), and Infineon shall have no liability arising out of any Security Breaches. Unless otherwise
# explicitly approved by Infineon, the Software may not be used in any application where a failure of the Product or any
# consequences of the use thereof can reasonably be expected to result in personal injury.

import os
import shutil
import sys
import tempfile

# Make the bundled pipeline package importable when launched from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_file, url_for

from model_converter import (
    ModelConverter,
    resolve_targets,
    is_compare_spec,
)

# Root under which generated artifacts live and from which downloads are served.
BASE_DIR = os.environ.get("ZOO_BASE_DIR", "/home/ubuntu")
OUT_DIR = os.path.join(BASE_DIR, "out")

app = Flask(__name__)
try:
    app.config["MAX_CONTENT_LENGTH"] = int(
        os.environ.get("ZOO_MAX_CONTENT_LENGTH", 512 * 1024 * 1024)
    )
except ValueError as exc:
    raise RuntimeError("ZOO_MAX_CONTENT_LENGTH must be an integer") from exc


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "request exceeds the configured upload limit"}), 413


def _as_bool(value) -> bool:
    """Interpret a form value as a boolean flag."""
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _download_url(abs_path: str):
    """Build a download URL for a generated file, or None if it doesn't exist."""
    if not os.path.isfile(abs_path):
        return None
    rel = os.path.relpath(abs_path, BASE_DIR)
    return url_for("download_file", filename=rel, _external=True)


def _single_target_files(target_out: str) -> dict:
    """Map the artifacts produced for a single target to download URLs."""
    candidates = {
        "model.c": "model.c",  # generated NN C code (for integration)
        "main.c": "main.c",  # test harness with embedded data
        "model.elf": "model.elf",  # compiled ELF
        "model.md": "model.md",  # onnx2c conversion report
        "results.json": "results.json",  # harmonized result table
        "pipeline.log": "pipeline.log",  # full pipeline log
        "model.tsim_prof.log": "model.tsim_prof.log",  # only present on TSIM runs
    }
    files = {}
    for label, fname in candidates.items():
        url = _download_url(os.path.join(target_out, fname))
        if url:
            files[label] = url
    return files


@app.route("/convert", methods=["GET", "POST"])
def convert():
    """Convert an ONNX model to C, compile, and benchmark on one or more targets.

    Form fields (POST):
      - ``onnx-file`` : the ONNX model
      - ``input_0``   : reference input tensor (protobuf)
      - ``output_0``  : reference output tensor (protobuf)
      - ``target``    : a single target (``tc3xx``, ``tc4dx``, ``arm_m4``,
                        ``tc4dx_ppu``; legacy ``TC3``/``TC4`` also accepted), a
                        space/comma separated list, or the keyword ``compare``/
                        ``all`` to benchmark every platform
      - ``tsim``      : optional; selects the TriCore timing backend. TSIM is the
                        default — omit the field or send a truthy value for TSIM,
                        send a falsy value to use the QEMU+CPI alternative.
                        Ignored for ARM/PPU targets.
      - ``profile``   : optional, enable per-node instruction profiling

    Returns a JSON map of artifact name -> download URL. For a single target
    this includes the generated ``model.c``; for a comparison it includes the
    aggregated ``comparison.txt``/``comparison.json`` plus per-target results.
    """
    if request.method != "POST":
        return "Welcome to the conversion server"

    # --- Receive the uploaded model + reference data ---
    missing = [
        name
        for name in ("onnx-file", "input_0", "output_0")
        if name not in request.files or not request.files[name].filename
    ]
    if missing:
        return jsonify({"error": "missing required upload(s)", "fields": missing}), 400

    request_dir = tempfile.mkdtemp(prefix="request-", dir=OUT_DIR)
    work_dir = os.path.join(request_dir, "_input")
    test_data_dir = os.path.join(work_dir, "test_data_set")
    os.makedirs(test_data_dir, exist_ok=True)

    onnx_path = os.path.join(work_dir, "model.onnx")
    request.files["onnx-file"].save(onnx_path)
    request.files["input_0"].save(os.path.join(test_data_dir, "input_0.pb"))
    request.files["output_0"].save(os.path.join(test_data_dir, "output_0.pb"))

    try:
        target_spec = request.form.get("target", "tc3xx")
        # TSIM is the default TriCore timing backend. The legacy `tsim` form field is
        # kept for backward compatibility: absent or truthy selects TSIM, an explicit
        # falsy value selects the QEMU+CPI alternative. Ignored for ARM/PPU.
        tsim_field = request.form.get("tsim")
        profiler = "tsim" if tsim_field is None or _as_bool(tsim_field) else "qemu"
        use_profile = _as_bool(request.form.get("profile", "false"))
        targets = resolve_targets(target_spec)
    except ValueError as exc:
        shutil.rmtree(request_dir, ignore_errors=True)
        return jsonify({"error": str(exc)}), 400

    converter = ModelConverter(
        model=onnx_path,
        test_data=test_data_dir,
        out_dir=request_dir,
        profile=use_profile,
        profiler=profiler,
    )

    # --- Comparison across multiple platforms ---
    if is_compare_spec(target_spec):
        try:
            comparison = converter.compare(targets)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
        files = {}
        comp_txt = _download_url(os.path.join(request_dir, "comparison.txt"))
        comp_json = _download_url(os.path.join(request_dir, "comparison.json"))
        if comp_txt:
            files["comparison.txt"] = comp_txt
        if comp_json:
            files["comparison.json"] = comp_json
        # Per-target details (results + log + generated C code)
        for target in targets:
            target_out = os.path.join(request_dir, target)
            for label, url in _single_target_files(target_out).items():
                files[f"{target}/{label}"] = url
        comparison_status = comparison.get("status", "unknown")
        status_code = 200 if comparison_status == "ok" else 422
        return jsonify({"status": comparison_status, "artifacts": files}), status_code

    # --- Single target conversion ---
    target = targets[0]
    try:
        result = converter.run(target)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    target_out = os.path.join(request_dir, target)
    status_code = 200 if result.status == "ok" else 422
    return (
        jsonify(
            {
                "status": result.status,
                "artifacts": _single_target_files(target_out),
            }
        ),
        status_code,
    )


@app.route("/download/<path:filename>", methods=["GET"])
def download_file(filename):
    """Serve a generated file as a download (paths are relative to BASE_DIR)."""
    # Prevent path traversal outside BASE_DIR.
    base_real = os.path.realpath(BASE_DIR)
    file_path = os.path.realpath(os.path.join(base_real, filename))
    if file_path != base_real and not file_path.startswith(base_real + os.sep):
        return "Invalid path", 400
    if not os.path.isfile(file_path):
        return "File not found", 404
    return send_file(file_path, mimetype="application/octet-stream", as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
