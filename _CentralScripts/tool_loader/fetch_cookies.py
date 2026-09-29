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
Extract Infineon Developer Center cookies and write them in Netscape format,
suitable for use with curl -b / -c.

Two strategies, tried in order:
  1. browser_cookie3  — silent; reads cookies from an already-logged-in Linux
                        browser (Chrome, Chromium, Firefox, Edge-for-Linux, …).
  2. Playwright       — interactive fallback; opens a headed Chromium window so
                        the user can log in, then captures the resulting cookies.

Works on native Linux and on WSL (WSLg provides the display for headed mode).

Usage:
    python3 fetch_cookies.py [OUTPUT_FILE]

Default output: infineon.cookies

Environment variables:
    LOGIN_TIMEOUT (seconds): Timeout for completing login and receiving an
                             access token (default: 180s)
"""

import json
import os
import sys
import time
from typing import Optional, List, Any, Dict, Tuple
from urllib.parse import urlparse

from security_utils import SESSION_HOSTS, atomic_write, parse_allowed_url

SEARCH_DOMAINS = (
    "softwaretools.infineon.com",
    "softwaretools-hosting.infineon.com",
)
LOGIN_URL = "https://softwaretools.infineon.com"
# Also visit the hosting domain so its session cookies are captured too.
HOSTING_URL = "https://softwaretools-hosting.infineon.com"
# Allow override via environment variable (in seconds)
LOGIN_TIMEOUT_SECS = int(os.environ.get("LOGIN_TIMEOUT", "180"))
OIDC_USER_KEY_PREFIX = "oidc.user:"
LOGIN_FAILURE_PATHS = frozenset(
    {"/auth/callback-error", "/forbidden", "/user-not-verified"}
)

BROWSERS = [
    "chrome",
    "chromium",
    "firefox",
    "edge",
    "brave",
    "opera",
    "vivaldi",
]


class _DictCookie:
    """Minimal cookie object wrapping a plain dict (Playwright cookie format)."""

    __slots__ = ("domain", "path", "secure", "expires", "name", "value")

    def __init__(self, d: Dict[str, Any]) -> None:
        self.domain = d.get("domain", "")
        self.path = d.get("path", "/")
        self.secure = bool(d.get("secure", False))
        self.expires = int(d.get("expires") or 0)
        self.name = d.get("name", "")
        self.value = d.get("value", "")


def _authorization_from_storage_state(state: Dict[str, Any]) -> Optional[str]:
    """Return a validated Bearer header from an OIDC user storage entry."""
    for origin in state.get("origins", []):
        if origin.get("origin") != LOGIN_URL:
            continue
        for item in origin.get("localStorage", []):
            if not item.get("name", "").startswith(OIDC_USER_KEY_PREFIX):
                continue
            try:
                oidc_user = json.loads(item.get("value", ""))
            except (TypeError, json.JSONDecodeError):
                continue
            token = oidc_user.get("access_token")
            token_type = oidc_user.get("token_type")
            if (
                isinstance(token, str)
                and token
                and "\r" not in token
                and "\n" not in token
                and isinstance(token_type, str)
                and token_type.lower() == "bearer"
            ):
                return f"Bearer {token}"
    return None


def _safe_page_url(url: str) -> str:
    """Strip query parameters and fragments that may contain OAuth secrets."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def _authorization_from_page(page: Any) -> Optional[str]:
    """Read OIDC authorization from the current page without opening another tab."""
    parsed = urlparse(page.url)
    if f"{parsed.scheme}://{parsed.netloc}" != LOGIN_URL:
        return None
    local_storage = page.evaluate(
        """() => Object.entries(window.localStorage).map(([name, value]) => ({name, value}))"""
    )
    state = {"origins": [{"origin": LOGIN_URL, "localStorage": local_storage}]}
    return _authorization_from_storage_state(state)


def _latest_open_page(context: Any) -> Optional[Any]:
    """Return the newest open page, following login redirects and popups."""
    open_pages = [page for page in context.pages if not page.is_closed()]
    return open_pages[-1] if open_pages else None


def _wait_for_oidc_authorization(
    context: Any,
) -> Tuple[Optional[str], Optional[Any]]:
    """Wait until callback processing stores a usable OIDC access token."""
    deadline = time.monotonic() + LOGIN_TIMEOUT_SECS
    while time.monotonic() < deadline:
        page = _latest_open_page(context)
        if page is None:
            print("Browser was closed before login completed.", file=sys.stderr)
            return None, None

        authorization = _authorization_from_page(page)
        if authorization:
            return authorization, page

        current_path = urlparse(page.url).path
        if current_path in LOGIN_FAILURE_PATHS:
            print(
                f"Login failed at {_safe_page_url(page.url)}.",
                file=sys.stderr,
            )
            return None, page
        page.wait_for_timeout(250)

    page = _latest_open_page(context)
    final_url = _safe_page_url(page.url) if page else "<browser closed>"
    print(
        f"Login did not produce an access token within {LOGIN_TIMEOUT_SECS}s. "
        f"Final page: {final_url}",
        file=sys.stderr,
    )
    return None, page


def _get_cookies_browser_cookie3() -> Optional[List[Any]]:
    """Try to read existing session cookies from locally installed Linux browsers."""
    try:
        import browser_cookie3
    except ImportError:
        return None  # not installed; caller decides what to do

    cookies: List[Any] = []
    found: List[str] = []
    for name in BROWSERS:
        fn = getattr(browser_cookie3, name, None)
        if fn is None:
            continue
        try:
            batch = []
            for domain in SEARCH_DOMAINS:
                batch.extend(fn(domain_name=domain))
            if batch:
                found.append(name)
                cookies.extend(batch)
        except Exception as e:
            print(
                f"Warning: Failed to extract cookies from {name}: {e}", file=sys.stderr
            )

    if found:
        print(f"Found cookies from: {', '.join(found)}", file=sys.stderr)

    return cookies


def _get_cookies_playwright(
    auth_file: Optional[str] = None, session_file: Optional[str] = None
) -> List[Any]:
    """Open a headed browser, complete OIDC login, and return captured cookies."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright is not installed.\n"
            "Install it with:\n"
            "  pip install playwright\n"
            "  playwright install chromium",
            file=sys.stderr,
        )
        return []

    print(
        f"Opening browser: {LOGIN_URL}\n"
        "Log in to Infineon Developer Center. "
        "The browser will close automatically once login is complete.",
        file=sys.stderr,
    )

    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=False)
            except Exception as e:
                print(f"Failed to launch Chromium: {e}", file=sys.stderr)
                print("Run:  playwright install chromium", file=sys.stderr)
                return []

            context = browser.new_context()

            # Intercept API requests to capture the Bearer token stored in localStorage.
            captured_auth: Dict[str, str] = {}

            def on_request(request: Any) -> None:
                auth = request.headers.get("authorization", "")
                if not auth.startswith("Bearer "):
                    return
                try:
                    parse_allowed_url(request.url, SESSION_HOSTS)
                except ValueError:
                    return
                captured_auth["authorization"] = auth

            context.on("request", on_request)

            page = context.new_page()
            page.goto(LOGIN_URL)

            authorization, page = _wait_for_oidc_authorization(context)
            if not authorization:
                try:
                    browser.close()
                except Exception:
                    pass
                return []
            print("Login completed and access token received.", file=sys.stderr)

            # Snapshot cookies after main-site auth.
            all_cookies: List[Any] = []
            try:
                all_cookies = context.cookies()
            except Exception as e:
                print(f"Warning: Failed to retrieve cookies: {e}", file=sys.stderr)

            # Navigate to the software listing — this triggers API calls that carry the Bearer token.
            try:
                page.goto(f"{LOGIN_URL}/assets/software", timeout=15000)
                page.wait_for_load_state("networkidle", timeout=20000)
                all_cookies = context.cookies()
            except Exception as e:
                print(
                    f"Warning: Could not navigate to assets page: {e}", file=sys.stderr
                )
                pass  # use snapshot from before assets navigation

            # Save full browser state (cookies + localStorage) for Playwright-based downloads.
            if session_file:
                try:
                    old_umask = os.umask(0o077)
                    try:
                        context.storage_state(path=session_file)
                    finally:
                        os.umask(old_umask)
                    os.chmod(session_file, 0o600)
                    print(f"Saved browser session to {session_file}", file=sys.stderr)
                except Exception as _e:
                    print(f"Could not save browser session: {_e}", file=sys.stderr)

            authorization = captured_auth.get("authorization", authorization)
            if auth_file:
                try:
                    atomic_write(auth_file, authorization.encode("utf-8"))
                    print(f"Saved auth token to {auth_file}", file=sys.stderr)
                except Exception as _e:
                    print(f"Could not save auth token: {_e}", file=sys.stderr)

            try:
                browser.close()
            except Exception:
                pass

        return [
            _DictCookie(c)
            for c in all_cookies
            if any(
                c.get("domain", "").lstrip(".") == domain for domain in SEARCH_DOMAINS
            )
            and c.get("name")
        ]
    except Exception as e:
        print(f"Playwright error: {e}", file=sys.stderr)
        return []


def get_cookies(auth_file=None, session_file=None):
    # --- Strategy 1: silent read from an existing Linux browser session ---
    cookies = _get_cookies_browser_cookie3()
    if cookies is None:
        print(
            "browser_cookie3 is not installed (silent strategy unavailable).\n"
            "Falling back to interactive login via Playwright.",
            file=sys.stderr,
        )
    elif cookies:
        return cookies
    else:
        print(
            "No Infineon cookies found in local Linux browsers.\n"
            "Falling back to interactive login via Playwright.",
            file=sys.stderr,
        )

    # --- Strategy 2: interactive Playwright login ---
    return _get_cookies_playwright(auth_file=auth_file, session_file=session_file)


def write_netscape(cookies: List[Any], outfile: str) -> None:
    """Write cookies in Netscape format (for curl -b / -c)."""
    lines = ["# Netscape HTTP Cookie File\n", "# Generated by fetch_cookies.py\n"]
    for c in cookies:
        domain = c.domain or ""
        include_subdomains = "TRUE" if domain.startswith(".") else "FALSE"
        secure = "TRUE" if c.secure else "FALSE"
        expires = int(c.expires) if c.expires else 0
        lines.append(
            f"{domain}\t{include_subdomains}\t{c.path}\t{secure}\t{expires}\t{c.name}\t{c.value}\n"
        )
    atomic_write(outfile, "".join(lines).encode("utf-8"))


def main() -> None:
    """Main entry point."""
    outfile = sys.argv[1] if len(sys.argv) > 1 else "infineon.cookies"
    auth_file = os.path.splitext(outfile)[0] + ".auth"
    session_file = os.path.splitext(outfile)[0] + ".session.json"

    cookies = get_cookies(auth_file=auth_file, session_file=session_file)
    if not cookies:
        print(
            f"No cookies found for {', '.join(SEARCH_DOMAINS)}.\n"
            "Make sure you are logged in to the Infineon Developer Center\n"
            f"({LOGIN_URL}) in your browser, then retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    write_netscape(cookies, outfile)
    print(f"Wrote {len(cookies)} cookie(s) to {outfile}")


if __name__ == "__main__":
    main()
