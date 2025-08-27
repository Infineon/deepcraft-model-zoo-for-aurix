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


RTOL = 1e-3
RTOL_LIMIT = 1e-3
ATOL = 1e-3
ATOL_LIMIT = 1e10
TEST_DATA_SET = 0

# Base directory for all tools and files
BASE_DIR = "/home/ubuntu"

CONVERSION_LOG_FILE = f"{BASE_DIR}/model_conversion.log"

ONNX2C = f"{BASE_DIR}/onnx2c_model_zoo"
TESTGEN_TC3 = f"{BASE_DIR}/testgen_target_tc3x_model_zoo"
TESTGEN_TC4 = f"{BASE_DIR}/testgen_target_tc4x_model_zoo"

TC_GCC = f"{BASE_DIR}/tricore_gcc/bin/tricore-elf-gcc"

TC_LINKER = f"{BASE_DIR}/TC_compilation/tsim-lto.ld"
TC_OBJECT = f"{BASE_DIR}/TC_compilation/crttsim.o"

QEMU = f"{BASE_DIR}/qemu-system-tricore"
QEMU_LOG = f"{BASE_DIR}/qemu.log"

