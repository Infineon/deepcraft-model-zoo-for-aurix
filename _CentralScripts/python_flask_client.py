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


import json
from pathlib import Path
from typing import Union

import requests


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
        target (str): Target platform for the model conversion.
    """

    def __init__(
        self,
        folder: Union[str, Path],
        target: str,
        url: str = "http://localhost:8080/convert",
    ):
        """
        Initialize the CallTools instance with the given folder, target, and URL.

        Args:
            folder (str or Path): Path to the folder containing the model and input/output files.
            target (str): Target platform for the model conversion.
            url (str): URL of the tool server (default is "http://localhost:8080/convert").
        """
        folder = Path(folder)
        assert folder.exists(), f"folder {folder} does not exist"
        self.folder = folder

        self.onnx_file = next(folder.rglob("model.onnx"), None)
        if not self.onnx_file:
            print(f"onnx file not found in {folder}")
        else:
            assert self.onnx_file.exists(), f"onnx file {self.onnx_file} does not exist"

        self.input = next(folder.rglob("input_0.pb"), None)
        if not self.input:
            print(f"input file not found in {folder}")
        else:
            assert self.input.exists(), f"input file {self.input} does not exist"

        self.output = next(folder.rglob("output_0.pb"), None)
        if not self.output:
            print(f"output file not found in {folder}")
        else:
            assert self.output.exists(), f"output file {self.output} does not exist"

        self.target_folder = folder / target
        if self.target_folder.exists():
            print(f"Target folder {self.target_folder} already exists")
        self.target_folder.mkdir(exist_ok=True)

        try:
            response = requests.get(url, timeout=100)
            response.raise_for_status()
            self.url = url
        except requests.RequestException as e:
            print(f"Error reaching the URL {url}: {e}")

        if target not in ["TC3", "TC4"]:
            raise ValueError(
                f"Invalid target '{target}'. Must be one of ['TC3', 'TC4']."
            )
        self.target = target

    def convert_model(self):
        """
        Convert the ONNX model using the remote tool server and download the converted files.

        The method uploads the ONNX model, input, and output files to the tool server,
        and then downloads the converted files to the target folder.
        """
        with open(self.onnx_file, "rb") as f1, open(self.input, "rb") as f2, open(
            self.output, "rb"
        ) as f3:
            file = {
                "onnx-file": f1,
                "input_0": f2,
                "output_0": f3,
            }
            data = {"target": self.target}

            response = requests.post(self.url, files=file, data=data, timeout=200)

        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            return
        elif not self.target_folder.exists():
            print(f"Error: Target folder {self.target_folder} does not exist")
            return
        else:
            file_urls = json.loads(response.content)

            downloaded_files = {}
            for filename, url in file_urls.items():
                response = requests.get(url, stream=True, timeout=200)
                if response.status_code == 200:
                    with open(self.target_folder / filename, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            f.write(chunk)
                    downloaded_files[filename] = True
                else:
                    downloaded_files[filename] = False
            print(downloaded_files)
