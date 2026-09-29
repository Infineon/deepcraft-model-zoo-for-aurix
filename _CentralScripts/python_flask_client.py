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
from typing import Union

import requests

# Canonical hardware targets exposed by the conversion service.
VALID_TARGETS = {"tc3xx", "tc4dx", "arm_m4", "tc4dx_ppu"}
# Legacy aliases accepted for backwards compatibility.
TARGET_ALIASES = {"tc3", "tc4", "ppu", "armm4"}
# Keywords that request a cross-platform comparison over every target.
COMPARE_KEYWORDS = {"compare", "all"}


class CallTools:
    """
    A class to handle the conversion of ONNX models using a remote tool server.

    Attributes:
        folder (Path): Path to the folder containing the model and input/output files.
        onnx_file (Path): Path to the ONNX model file.
        input (Path): Path to the input file.
        output (Path): Path to the output file.
        target_folder (Path): Path to the target folder for storing converted files.
        url (str): URL of the tool server.
        target (str): Target platform(s) for the model conversion. A single
            target (e.g. ``"tc4dx"`` or legacy ``"TC4"``), a space/comma
            separated list, or the keyword ``"compare"``/``"all"`` to benchmark
            the model on every supported platform.
        tsim (bool): TriCore timing backend selector. TSIM is the default;
            set ``tsim=False`` to use the QEMU+CPI alternative instead
            (TriCore targets only; ignored for ARM/PPU).
        profile (bool): Enable per-node instruction profiling.
    """

    def __init__(
        self,
        folder: Union[str, Path],
        target: str,
        url: str = "http://localhost:8080/convert",
        tsim: bool = True,
        profile: bool = False,
    ):
        """
        Initialize the CallTools instance with the given folder, target, and URL.

        Args:
            folder (str or Path): Path to the folder containing the model and input/output files.
            target (str): Target platform(s) for the model conversion (see class docstring).
            url (str): URL of the tool server (default is "http://localhost:8080/convert").
            tsim (bool): TriCore timing backend. TSIM is the default; set
                ``tsim=False`` to use QEMU+CPI instead (TriCore targets only).
            profile (bool): Enable per-node instruction profiling.
        """
        folder = Path(folder)
        assert folder.exists(), f"folder {folder} does not exist"
        self.folder = folder

        self.onnx_file = next(folder.rglob("model.onnx"), None)
        if not self.onnx_file:
            raise FileNotFoundError(f"onnx file not found in {folder}")
        else:
            assert self.onnx_file.exists(), f"onnx file {self.onnx_file} does not exist"

        self.input = next(folder.rglob("input_0.pb"), None)
        if not self.input:
            raise FileNotFoundError(f"input file not found in {folder}")
        else:
            assert self.input.exists(), f"input file {self.input} does not exist"

        self.output = next(folder.rglob("output_0.pb"), None)
        if not self.output:
            raise FileNotFoundError(f"output file not found in {folder}")
        else:
            assert self.output.exists(), f"output file {self.output} does not exist"

        self._validate_target(target)
        self.target = self._canonical_target_spec(target)
        self.tsim = tsim
        self.profile = profile
        self.is_compare = self._is_compare(self.target)

        # Downloads are organized in per-hardware subfolders, e.g.
        # ``<folder>/tc4dx/model.c``. In comparison mode the server returns keys
        # already prefixed with the target name ("tc4dx/model.c", ...) plus the
        # aggregated "comparison.txt"/"comparison.json", so the base folder is
        # the model folder itself. In single-target mode the bare filenames are
        # placed under ``<folder>/<target>/``.
        self.target_folder = folder if self.is_compare else folder / self.target
        if self.target_folder.exists() and not self.is_compare:
            print(f"Target folder {self.target_folder} already exists")
        self.target_folder.mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(url, timeout=100)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ConnectionError(f"Error reaching the URL {url}: {e}") from e
        self.url = url

    @staticmethod
    def _tokens(target: str):
        return [t for t in str(target).replace(",", " ").split() if t]

    @classmethod
    def _is_compare(cls, target: str) -> bool:
        tokens = cls._tokens(target)
        if len(tokens) == 1 and tokens[0].lower() in COMPARE_KEYWORDS:
            return True
        return len(tokens) > 1

    @classmethod
    def _validate_target(cls, target: str) -> None:
        tokens = cls._tokens(target)
        if not tokens:
            raise ValueError("No target specified.")
        if len(tokens) == 1 and tokens[0].lower() in COMPARE_KEYWORDS:
            return
        allowed = VALID_TARGETS | TARGET_ALIASES
        for tok in tokens:
            if tok.lower() not in allowed:
                raise ValueError(
                    f"Invalid target '{tok}'. Must be one of {sorted(VALID_TARGETS)} "
                    f"(aliases: TC3, TC4), a list, or 'compare'/'all'."
                )

    @classmethod
    def _canonical_target_spec(cls, target: str) -> str:
        tokens = cls._tokens(target)
        if len(tokens) == 1 and tokens[0].lower() in COMPARE_KEYWORDS:
            return tokens[0].lower()
        canonical = {
            "tc3": "tc3xx",
            "tc4": "tc4dx",
            "ppu": "tc4dx_ppu",
            "armm4": "arm_m4",
        }
        return " ".join(canonical.get(token.lower(), token.lower()) for token in tokens)

    def convert_model(self):
        """
        Convert the ONNX model using the remote tool server and download the converted files.

        The method uploads the ONNX model, input, and output files to the tool server,
        and then downloads the converted files to the target folder. Returns a dict
        mapping each downloaded artifact to a success flag.
        """
        # Comparison runs every platform sequentially, so allow more time.
        timeout = 3600 if self.is_compare else 900
        with open(self.onnx_file, "rb") as f1, open(self.input, "rb") as f2, open(
            self.output, "rb"
        ) as f3:
            file = {
                "onnx-file": f1,
                "input_0": f2,
                "output_0": f3,
            }
            data = {
                "target": self.target,
                "tsim": str(self.tsim).lower(),
                "profile": str(self.profile).lower(),
            }

            response = requests.post(self.url, files=file, data=data, timeout=timeout)

        if response.status_code not in (200, 422):
            print(
                f"Error: Received status code {response.status_code}: {response.text}"
            )
            return {}
        if not self.target_folder.exists():
            print(f"Error: Target folder {self.target_folder} does not exist")
            return {}

        try:
            response_data = response.json()
        except ValueError as exc:
            raise ValueError("Conversion service returned invalid JSON") from exc
        if not isinstance(response_data, dict):
            raise ValueError("Conversion service returned an invalid response")
        status = response_data.get("status", "ok")
        file_urls = response_data.get("artifacts", response_data)
        if not isinstance(file_urls, dict):
            raise ValueError("Conversion service returned invalid artifacts")

        downloaded_files = {}
        for filename, url in file_urls.items():
            dest = (self.target_folder / filename).resolve()
            target_root = self.target_folder.resolve()
            if dest != target_root and target_root not in dest.parents:
                raise ValueError(f"Unsafe artifact path: {filename}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                file_response = requests.get(url, stream=True, timeout=timeout)
                file_response.raise_for_status()
            except requests.RequestException as exc:
                downloaded_files[filename] = False
                print(f"Error downloading {filename}: {exc}")
                continue
            with open(dest, "wb") as f:
                for chunk in file_response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            downloaded_files[filename] = True
        if status != "ok":
            downloaded_files["__conversion_status__"] = False
        print(downloaded_files)
        return downloaded_files
