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

"""Shared security checks for authenticated tool downloads."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import ParseResult, urlparse

ALLOWED_DOWNLOAD_HOSTS = frozenset(
    {
        "softwaretools-hosting.infineon.com",
        "softwaretools.infineon.com",
        "softwaretools-preview.icp.infineon.com",
        "artifactory.intra.infineon.com",
    }
)
SESSION_HOSTS = frozenset(
    {
        "softwaretools-hosting.infineon.com",
        "softwaretools.infineon.com",
        "softwaretools-preview.icp.infineon.com",
    }
)
MIN_ARTIFACT_SIZE = 10 * 1024
_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


def _host_allowed(host: str, allowed_hosts: Iterable[str]) -> bool:
    host = host.lower().rstrip(".")
    return any(
        host == allowed or host.endswith("." + allowed) for allowed in allowed_hosts
    )


def parse_allowed_url(
    url: str, allowed_hosts: Iterable[str] = ALLOWED_DOWNLOAD_HOSTS
) -> ParseResult:
    """Parse and validate an HTTPS URL against an approved host set."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.username or parsed.password:
        raise ValueError("URL must use HTTPS and must not contain user information")
    if not parsed.hostname or not _host_allowed(parsed.hostname, allowed_hosts):
        raise ValueError(f"URL host is not approved: {parsed.hostname or '<missing>'}")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("URL contains an invalid port") from exc
    if port not in (None, 443):
        raise ValueError("URL must use the default HTTPS port")
    return parsed


def validate_filename(filename: str) -> None:
    """Allow only a single ordinary artifact filename."""
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or not _FILENAME_RE.fullmatch(filename)
    ):
        raise ValueError(f"invalid artifact filename: {filename!r}")


def validate_artifact(
    path: str, content_type: Optional[str] = None, sha256: str = ""
) -> None:
    """Validate size, HTML responses, and an optional SHA-256 checksum."""
    artifact = Path(path)
    size = artifact.stat().st_size
    if size < MIN_ARTIFACT_SIZE:
        raise ValueError(f"downloaded file is too small ({size} bytes)")
    if content_type and content_type.lower().startswith("text/html"):
        raise ValueError("response is HTML, not a package artifact")
    with artifact.open("rb") as stream:
        prefix = stream.read(512).lower()
    if b"<!doctype html" in prefix or b"<html" in prefix:
        raise ValueError("response is HTML, not a package artifact")
    if sha256:
        digest = hashlib.sha256()
        with artifact.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != sha256.lower():
            raise ValueError(f"SHA-256 mismatch: expected {sha256}, got {actual}")


def atomic_write(path: str, data: bytes) -> None:
    """Write sensitive data with mode 0600 and replace the target atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=str(destination.parent)
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
