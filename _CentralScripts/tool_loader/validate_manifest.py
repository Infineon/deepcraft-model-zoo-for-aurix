#!/usr/bin/env python3
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

"""Validate the tool manifest before the downloader performs network access."""

from __future__ import annotations

import csv
import re
import sys

from security_utils import parse_allowed_url, validate_filename

REQUIRED_COLUMNS = [
    "tool_id",
    "version",
    "platform",
    "filename",
    "url",
    "auth_required",
    "login_required",
    "sha256",
]
BOOLEAN_COLUMNS = {"auth_required", "login_required"}
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def validate_manifest(path: str) -> None:
    seen_tools = set()
    seen_files = set()
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != REQUIRED_COLUMNS:
            raise ValueError(
                f"expected columns {REQUIRED_COLUMNS}, got {reader.fieldnames}"
            )
        for row_number, row in enumerate(reader, start=2):
            if not row.get("tool_id") or row["tool_id"].startswith("#"):
                continue
            missing = [column for column in REQUIRED_COLUMNS if row.get(column) is None]
            if missing:
                raise ValueError(
                    f"row {row_number}: missing columns: {', '.join(missing)}"
                )
            for column in BOOLEAN_COLUMNS:
                if row[column].lower() not in {"true", "false"}:
                    raise ValueError(
                        f"row {row_number}: {column} must be true or false"
                    )
            if row["tool_id"] in seen_tools:
                raise ValueError(
                    f"row {row_number}: duplicate tool_id {row['tool_id']!r}"
                )
            if row["filename"] in seen_files:
                raise ValueError(
                    f"row {row_number}: duplicate filename {row['filename']!r}"
                )
            validate_filename(row["filename"])
            parse_allowed_url(row["url"])
            if row["sha256"] and not SHA256_RE.fullmatch(row["sha256"]):
                raise ValueError(
                    f"row {row_number}: sha256 must be 64 hexadecimal characters"
                )
            seen_tools.add(row["tool_id"])
            seen_files.add(row["filename"])


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} MANIFEST", file=sys.stderr)
        return 2
    try:
        validate_manifest(sys.argv[1])
    except (OSError, ValueError) as exc:
        print(f"Manifest validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
