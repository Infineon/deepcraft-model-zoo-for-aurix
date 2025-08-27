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

from flask import Flask, request, jsonify, send_file, url_for
from model_converter import ModelConverter
from config import BASE_DIR

app = Flask(__name__)


@app.route("/convert", methods=["GET", "POST"])
def convert():
    """
    Handle the conversion of an ONNX model to C code, compile it to an ELF file,
    and simulate it using QEMU or nSIM depending on target HW.
    Accepts ONNX model and input/output data files and a target parameter specifying the target hardware.
    via POST request and returns URLs to download the generated files.

    :return: JSON response with URLs to download the generated C file, ELF file, json results and log.
    """
    if request.method == "POST":
        # Receive the ONNX file
        test_folder = "model"
        test_subfolder = os.path.join(test_folder, "test_model/")
        test_data_folder = os.path.join(test_subfolder, "test_data_set_0/")
        if os.path.isdir(test_folder):
            shutil.rmtree(test_folder)
        os.mkdir(test_folder)
        os.mkdir(test_subfolder)
        os.mkdir(test_data_folder)

        onnx_model = "model.onnx"
        onnx_file = request.files["onnx-file"]
        onnx_file.save(test_subfolder + onnx_model)
        print("Received ONNX file")

        input_data = request.files["input_0"]
        input_data.save(test_data_folder + "input_0.pb")
        print("Received input data")

        output_data = request.files["output_0"]
        output_data.save(test_data_folder + "output_0.pb")
        print("Received output data")

        mconv = ModelConverter(
            model_path=test_subfolder,
            target=request.form["target"],
        )
        mconv.cleanup()
        mconv.generate_c_file()
        mconv.generate_testgen_file()
        mconv.compile_model()
        mconv.benchmark()

        # Remove the leading part '/home/ubuntu/' from the paths if present
        mconv.c_file = str(mconv.c_file).replace(BASE_DIR, "")
        mconv.testgen_file = str(mconv.testgen_file).replace(BASE_DIR, "")
        mconv.elf_file = str(mconv.elf_file).replace(BASE_DIR, "")
        mconv.conversion_log_file = str(mconv.conversion_log_file).replace(BASE_DIR, "")

        return jsonify(
            {
                "c_file.c": url_for(
                    "download_file", filename=mconv.c_file, _external=True
                ),
                "testgen.c": url_for(
                    "download_file", filename=mconv.testgen_file, _external=True
                ),
                "out.elf": url_for(
                    "download_file", filename=mconv.elf_file, _external=True
                ),
                "model_conversion.log": url_for(
                    "download_file", filename=mconv.conversion_log_file, _external=True
                ),
            }
        )

    else:
        return "Welcome to the conversion server"


@app.route("/download/<path:filename>", methods=["GET", "POST"])
def download_file(filename):
    """
    Handle the download of generated files.

    :param filename: Name of the file to be downloaded.
    :return: The requested file as an attachment.
    """
    if request.method == "GET":
        file_path = os.path.join("/home/ubuntu/", filename)
        return send_file(
            file_path, mimetype="application/octet-stream", as_attachment=True
        )

    else:
        return "Welcome to the download server"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
