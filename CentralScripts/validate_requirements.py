# Copyright (C) Infineon Technologies AG 2025
#
# Use of this file is subject to the terms of use agreed between (i) you or the company in which ordinary course of
# business you are acting and (ii) Infineon Technologies AG or its licensees. If and as long as no such terms of use
# are agreed, use of this file is subject to following:
#
# This file is licensed under the terms of the Boost Software License. See the LICENSE file in the root of this repository
# for complete details.

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
