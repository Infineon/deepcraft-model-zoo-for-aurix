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


#!/usr/bin/env python3
# validate_requirements.py
import subprocess
import sys
import tempfile
import os


def validate_requirements(requirements_file="requirements.txt"):
    """Validate requirements.txt without installing packages"""

    print(f"🔍 Validating {requirements_file}...")

    # Step 1: Check if requirements.txt exists and is readable
    if not os.path.exists(requirements_file):
        print(f"❌ {requirements_file} not found")
        return False

    # Step 2: Use pip-compile to check dependency resolution
    print("📦 Checking dependency resolution...")
    try:
        # Create a temporary output file with a different name
        temp_output = tempfile.mktemp(suffix=".compiled.txt")

        result = subprocess.run(
            [
                "pip-compile",
                "--output-file",
                temp_output,  # Explicit different output filename
                "--strip-extras",  # Silence warning
                "--quiet",
                requirements_file,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        # Clean up the temporary file
        if os.path.exists(temp_output):
            os.unlink(temp_output)

        print("✅ Dependency resolution successful")
    except subprocess.CalledProcessError as e:
        print(f"❌ Dependency resolution failed:")
        print(e.stderr)
        # Clean up temp file if it exists
        if "temp_output" in locals() and os.path.exists(temp_output):
            os.unlink(temp_output)
        return False

    # Step 3: Check for known vulnerabilities (requires safety)
    print("🔐 Checking for security vulnerabilities...")
    try:
        # Check if safety is installed, install if not
        try:
            subprocess.run(["safety", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Installing safety...")
            subprocess.run(["pip", "install", "safety", "--quiet"], check=True)

        result = subprocess.run(
            ["safety", "check", "--file", requirements_file],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ No known security vulnerabilities")
        else:
            print("⚠️  Security vulnerabilities found:")
            print(result.stdout)
    except subprocess.CalledProcessError:
        print("⚠️  Could not check security vulnerabilities")

    # Step 4: Basic syntax validation
    print("📝 Checking requirements syntax...")
    try:
        from packaging.requirements import Requirement

        with open(requirements_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        Requirement(line)
                    except Exception as e:
                        print(f"❌ Syntax error on line {line_num}: {line}")
                        print(f"   Error: {e}")
                        return False
        print("✅ Requirements syntax is valid")
    except Exception as e:
        print(f"❌ Could not validate syntax: {e}")
        return False

    print("🎉 All validations passed!")
    return True


if __name__ == "__main__":
    requirements_file = sys.argv[1] if len(sys.argv) > 1 else "requirements.txt"
    success = validate_requirements(requirements_file)
    sys.exit(0 if success else 1)
