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

import logging
import os
import subprocess

from config import *

logging.basicConfig(
    filename=CONVERSION_LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class ModelConverter:
    def __init__(self, model_path: str, target: str):
        assert os.path.exists(model_path), f"model path {model_path} does not exist"
        self.model_path: str = model_path

        self.target: str = target
        self.c_file: str = "out/model.c"
        self.testgen_file: str = "out/testgen.c"
        self.conversion_log_file: str = CONVERSION_LOG_FILE
        self.elf_file: str = "out/out.elf"
        self.test_results_json: str = "out/test_results.json"

        # Ensure the logfile is empty
        if os.path.exists(self.conversion_log_file):
            with open(self.conversion_log_file, "w", encoding="utf-8") as log_file:
                log_file.truncate(0)

        # set tolerances
        self.rtol: float = RTOL
        self.rtol_limit: float = RTOL_LIMIT
        self.atol: float = ATOL
        self.atol_limit: float = ATOL_LIMIT
        self.test_data_set: int = TEST_DATA_SET

        logging.info("ModelConverter initialized with model_path: %s", model_path)
        logging.info("Target: %s", target)
        logging.info("C file will be generated at: %s", self.c_file)

    def cleanup(self) -> None:
        """Remove all files from the 'out/' folder."""
        out_dir = "out"

        if not os.path.exists(out_dir):
            logging.info(
                "Output directory '%s' does not exist, nothing to clean up", out_dir
            )
            return

        if not os.path.isdir(out_dir):
            logging.warning("'%s' exists but is not a directory", out_dir)
            return

        files_deleted = 0
        for filename in os.listdir(out_dir):
            file_path = os.path.join(out_dir, filename)

            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    logging.info("Deleted file: %s", file_path)
                    files_deleted += 1
                except OSError as e:
                    logging.error("Failed to delete file %s: %s", file_path, e)
            else:
                logging.info("Skipping non-file item: %s", file_path)

        if files_deleted == 0:
            logging.info("No files found in '%s' folder to delete", out_dir)
        else:
            logging.info(
                "Cleanup completed: %d file(s) deleted from '%s' folder",
                files_deleted,
                out_dir,
            )

    def get_model_file(self) -> str:
        for root, _, files in os.walk(self.model_path):
            for file in files:
                if file == "model.onnx":
                    model_file_path = os.path.join(root, file)
                    logging.info("Found model file: %s", model_file_path)
                    return model_file_path
        raise FileNotFoundError(
            f"No model.onnx file found in {self.model_path} or its subfolders"
        )

    def generate_c_file(self) -> None:
        model_file = self.get_model_file()
        command = ""
        if self.target == "TC3" or self.target == "TC4":
            command = f"{ONNX2C} {model_file} -punionize -l3 > {self.c_file}"
        else:
            logging.error("Unknown target %s, select from ['TC3', 'TC4']", self.target)
            raise ValueError(
                f"Unknown target {self.target}, select from ['TC3', 'TC4']"
            )

        logging.info("Generating C file with command: %s", command)
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        logging.info(result.stdout.decode())

        if os.path.exists(self.c_file) and os.path.getsize(self.c_file) > 0:
            logging.info("C file generated successfully: %s", self.c_file)
        else:
            if not os.path.exists(self.c_file):
                logging.error(
                    "Failed to generate testgen file: %s (file does not exist)",
                    self.c_file,
                )
            elif os.path.getsize(self.c_file) == 0:
                logging.error(
                    "Failed to generate testgen file: %s (file is empty)", self.c_file
                )

    def generate_testgen_file(self) -> None:
        command = ""
        if self.target == "TC3":
            command = f"{TESTGEN_TC3} {self.model_path} {self.rtol} {self.atol} {self.rtol_limit} {self.atol_limit} {self.test_data_set} -punionize -l3 > {self.testgen_file}"
        elif self.target == "TC4":
            command = f"{TESTGEN_TC4} {self.model_path} {self.rtol} {self.atol} {self.rtol_limit} {self.atol_limit} {self.test_data_set} -punionize -l3 > {self.testgen_file}"
        else:
            logging.error("Unknown target %s, select from ['TC3', 'TC4']", self.target)
            raise ValueError(
                f"Unknown target {self.target}, select from ['TC3', 'TC4']"
            )

        logging.info("Generating testgen file with command: %s", command)
        logging.info(
            "rtol: %s, atol: %s, rtol_limit: %s, atol_limit: %s, test_data_set: %s",
            self.rtol,
            self.atol,
            self.rtol_limit,
            self.atol_limit,
            self.test_data_set,
        )
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        logging.info(result.stdout.decode())

        if os.path.exists(self.testgen_file) and os.path.getsize(self.testgen_file) > 0:
            logging.info("Testgen file generated successfully: %s", self.testgen_file)
        else:
            if not os.path.exists(self.testgen_file):
                logging.error(
                    "Failed to generate testgen file: %s (file does not exist)",
                    self.testgen_file,
                )
            elif os.path.getsize(self.testgen_file) == 0:
                logging.error(
                    "Failed to generate testgen file: %s (file is empty)",
                    self.testgen_file,
                )

    def compile_model(self) -> None:
        command = ""
        if self.target == "TC3":
            command = f"{TC_GCC} -Ofast -mcpu=tc39xx -save-temps -Wl,-gc-sections -Wl,--extmap=a -nocrt0 -mcpu=tc39xx -Xlinker --mcpu=tc162 -T{TC_LINKER} {TC_OBJECT} {self.testgen_file} -o {self.elf_file}"
        elif self.target == "TC4":
            command = f"{TC_GCC} -Ofast -mcpu=tc4DAx -save-temps -Wl,-gc-sections -Wl,--extmap=a -nocrt0 -mcpu=tc4DAx -Xlinker --mcpu=tc18 -T{TC_LINKER} {TC_OBJECT} {self.testgen_file} -o {self.elf_file}"
        else:
            logging.error("Unknown target %s, select from ['TC3', 'TC4']", self.target)
            raise ValueError(
                f"Unknown target {self.target}, select from ['TC3', 'TC4']"
            )

        logging.info("Compiling model with command: %s", command)
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        logging.info(result.stdout.decode())

        if os.path.exists(self.elf_file) and os.path.getsize(self.elf_file) > 0:
            logging.info("Model compiled successfully: %s", self.elf_file)
        else:
            if not os.path.exists(self.elf_file):
                logging.error(
                    "Failed to compile model: %s (file does not exist)", self.elf_file
                )
            elif os.path.getsize(self.elf_file) == 0:
                logging.error(
                    "Failed to compile model: %s (file is empty)", self.elf_file
                )

    def benchmark(self):
        command = ""
        if self.target == "TC3":
            command = f"{QEMU} -display none -M tricore_tsim162 -semihosting -kernel {self.elf_file} > QEMU.log"
        elif self.target == "TC4":
            command = f"{QEMU} -display none -M tricore_tsim18 -semihosting -kernel {self.elf_file} > QEMU.log"
        else:
            logging.error("Unknown target %s, select from ['TC3', 'TC4']", self.target)
            raise ValueError(
                f"Unknown target {self.target}, select from ['TC3', 'TC4']"
            )

        logging.info("Benchmarking model with command: %s", command)
        try:
            result = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=200,
            )
        except subprocess.TimeoutExpired:
            logging.error("Benchmarking command timed out and was interrupted.")
            return
        logging.info(result.stdout.decode())

        if os.path.exists("QEMU.log") and os.path.getsize("QEMU.log") > 0:
            logging.info(
                "Benchmarking completed successfully, results saved in QEMU.log"
            )
            # Add QEMU.log content to model_conversion.log
            with open("QEMU.log", "r", encoding="utf-8") as qemu_log_file:
                qemu_log_content = qemu_log_file.read()
                logging.info("QEMU.log content:\n%s", qemu_log_content)
        else:
            if not os.path.exists("QEMU.log"):
                logging.error(
                    "Failed to benchmark model: QEMU.log (file does not exist)"
                )
            elif os.path.getsize("QEMU.log") == 0:
                logging.error("Failed to benchmark model: QEMU.log (file is empty)")
