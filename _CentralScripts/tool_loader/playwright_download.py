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

"""
Download a file from the Infineon hosting backend using a saved Playwright
browser session (infineon.session.json).

The session is created by fetch_cookies.py and contains the full browser state
including localStorage tokens, so the app's JavaScript can authenticate the
download request transparently — no manual token extraction needed.

Usage:
    python3 playwright_download.py URL OUTPUT_PATH [SESSION_FILE] [SHA256]

Arguments:
    URL          The /packages/.../download URL from tools.csv.
    OUTPUT_PATH  Where to write the downloaded file.
    SESSION_FILE Path to the session state file (default: infineon.session.json
                 next to this script).
    SHA256       Optional expected SHA-256 checksum for integrity verification.

Environment variables:
    DOWNLOAD_TIMEOUT (seconds): Timeout for download completion (default: 600s)
"""

import os
import sys

from security_utils import parse_allowed_url, validate_artifact

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SESSION_FILE = os.path.join(SCRIPT_DIR, "infineon.session.json")
# Timeout for the download itself (allow override via environment variable in seconds)
try:
    DOWNLOAD_TIMEOUT_SECS = int(os.environ.get("DOWNLOAD_TIMEOUT", "600"))
except ValueError as exc:
    raise SystemExit("DOWNLOAD_TIMEOUT must be an integer") from exc
if DOWNLOAD_TIMEOUT_SECS <= 0:
    raise SystemExit("DOWNLOAD_TIMEOUT must be greater than zero")
DOWNLOAD_TIMEOUT_MS = DOWNLOAD_TIMEOUT_SECS * 1000
LOGIN_URL = "https://softwaretools.infineon.com"
MAX_RETRIES = 3


def download(
    url: str,
    output_path: str,
    session_file: str = DEFAULT_SESSION_FILE,
    sha256: str = "",
) -> None:
    """Download a file using Playwright with saved session and retry logic."""
    try:
        parse_allowed_url(url)
    except ValueError as exc:
        print(f"Rejected download URL: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        print(
            "playwright is not installed. Run: pip install playwright && playwright install chromium",
            file=sys.stderr,
        )
        sys.exit(2)

    if not os.path.exists(session_file):
        print(
            f"Session file not found: {session_file}\n"
            "Run: ./_CentralScripts/tool_loader/tool_loader.sh --auto-cookies to create it.",
            file=sys.stderr,
        )
        sys.exit(1)
    if os.stat(session_file).st_mode & 0o077:
        print(
            f"Session file is readable by group or others: {session_file}",
            file=sys.stderr,
        )
        sys.exit(1)

    # The hosting frontend /packages/.../download serves an HTML app shell that
    # triggers the real download via JavaScript once the app is authenticated.
    # We navigate there inside the saved session so the app has full auth state.
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(storage_state=session_file)
                page = context.new_page()

                print(
                    f"Downloading {url} (attempt {attempt}/{MAX_RETRIES})",
                    file=sys.stderr,
                )
                try:
                    with page.expect_download(timeout=DOWNLOAD_TIMEOUT_MS) as dl_info:
                        try:
                            page.goto(url, wait_until="commit", timeout=30000)
                        except Exception:
                            pass  # navigation "failure" is normal when a download starts
                    dl = dl_info.value
                    # The artifact may be delivered from a pre-signed redirect
                    # target (e.g. S3); integrity is enforced by SHA-256 below.
                    dl.save_as(output_path)
                    try:
                        validate_artifact(output_path, sha256=sha256)
                    except (OSError, ValueError) as exc:
                        try:
                            os.unlink(output_path)
                        except FileNotFoundError:
                            pass
                        # Validation failures (bad checksum/HTML/size) are deterministic; do not retry.
                        print(f"Download validation failed: {exc}", file=sys.stderr)
                        browser.close()
                        sys.exit(1)
                    print(f"Saved to {output_path}", file=sys.stderr)
                    browser.close()
                    return  # Success!
                except PWTimeout:
                    print(
                        f"Download timed out after {DOWNLOAD_TIMEOUT_SECS}s (attempt {attempt}/{MAX_RETRIES}).\n"
                        "The session may have expired — re-run with --auto-cookies.",
                        file=sys.stderr,
                    )
                    browser.close()
                    if attempt < MAX_RETRIES:
                        print("Retrying...", file=sys.stderr)
                        continue
                    sys.exit(1)
                except Exception as e:
                    print(
                        f"Download error (attempt {attempt}/{MAX_RETRIES}): {e}",
                        file=sys.stderr,
                    )
                    browser.close()
                    if attempt < MAX_RETRIES:
                        print("Retrying...", file=sys.stderr)
                        continue
                    sys.exit(1)
        except Exception as e:
            print(
                f"Playwright initialization error (attempt {attempt}/{MAX_RETRIES}): {e}",
                file=sys.stderr,
            )
            if attempt < MAX_RETRIES:
                print("Retrying...", file=sys.stderr)
                continue
            sys.exit(1)

    # Should not reach here, but just in case
    print(f"Failed to download after {MAX_RETRIES} attempts.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} URL OUTPUT_PATH [SESSION_FILE]", file=sys.stderr)
        sys.exit(1)

    _url: str = sys.argv[1]
    _out: str = sys.argv[2]
    _session: str = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_SESSION_FILE
    _sha256: str = sys.argv[4] if len(sys.argv) > 4 else ""

    download(_url, _out, _session, _sha256)
