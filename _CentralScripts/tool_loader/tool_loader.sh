#!/usr/bin/env bash

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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/downloads"
MANIFEST="$SCRIPT_DIR/tools.csv"
DEFAULT_COOKIE_FILE="$SCRIPT_DIR/infineon.cookies"
DEFAULT_AUTH_FILE="$SCRIPT_DIR/infineon.auth"
DEFAULT_SESSION_FILE="$SCRIPT_DIR/infineon.session.json"

_CURRENT_TMP=""
_CURRENT_HEADERS=""
_AUTH_CONFIG=""

_log_verbose() {
	if [[ $verbose -eq 1 ]]; then
		echo "[DEBUG] $*" >&2
	fi
}

_cleanup_tmp() {
	rm -f "$_CURRENT_TMP" "$_CURRENT_HEADERS" "$_AUTH_CONFIG"
}
trap _cleanup_tmp EXIT

# Store the Authorization header in a private curl config (-K) so the Bearer
# token is never exposed in the process list via argv.
_set_auth_config() {
	local auth_val="$1"
	if [[ -z "$auth_val" ]]; then
		CURL_AUTH_ARGS=()
		return
	fi
	[[ -n "$_AUTH_CONFIG" ]] || _AUTH_CONFIG="$(mktemp)"
	printf 'header = "Authorization: %s"\n' "$auth_val" > "$_AUTH_CONFIG"
	chmod 600 "$_AUTH_CONFIG"
	CURL_AUTH_ARGS=(-K "$_AUTH_CONFIG")
}

# Print an error and return non-zero if a file is readable by group or others.
_require_private_file() {
	local path="$1" label="$2"
	if (( 0$(stat -c '%a' "$path") & 077 )); then
		echo "$label is readable by group or others: $path" >&2
		return 1
	fi
}

# Validate a downloaded artifact (size, HTML sniff, optional SHA-256) via the
# shared Python implementation so curl and Playwright paths behave identically.
_validate_artifact() {
	PYTHONPATH="$SCRIPT_DIR" "$PYTHON" - "$1" "$2" <<'PY'
import sys
from security_utils import validate_artifact

try:
    validate_artifact(sys.argv[1], sha256=sys.argv[2])
except (OSError, ValueError) as exc:
    print(f"Validation failed: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

# Move tmp to final path and clear trap references. Callers validate beforehand.
_install_artifact() {
	local tool_id="$1" tmp_path="$2" out_path="$3"
	mv "$tmp_path" "$out_path"
	_CURRENT_TMP=""
	_CURRENT_HEADERS=""
	echo "Download finished: $out_path"
}

usage() {
	cat <<'EOF'
Usage: ./tool_loader.sh [OPTIONS]

Options:
	--tool TOOL_ID              Filter by tool ID (e.g., tsim, aurixgcc).
	--platform PLATFORM         Filter by platform (e.g., linux-x64).
	--cookie-file PATH          Path to cookie file (default: infineon.cookies).
	--auto-cookies              Extract cookies from the local browser automatically.
	--verbose, -v               Enable verbose output for debugging.
	--dry-run                   Show what would be downloaded without downloading.
	--help, -h                  Show this help message.

Examples:
	./tool_loader.sh
	./tool_loader.sh --auto-cookies
	./tool_loader.sh --tool tsim --auto-cookies
	./tool_loader.sh --platform linux-x64 --verbose
	./tool_loader.sh --dry-run
EOF
}

print_cookie_help() {
	cat >&2 <<EOF
Authentication required for this tool. Choose one of:

Option A – automatic (recommended):
  1) Run the model-zoo setup from the repository root:
	  ./_CentralScripts/setup.sh
	Setup installs the downloader dependencies and starts authentication.
  2) To retry authentication directly, run:
	  ./_CentralScripts/tool_loader/tool_loader.sh --auto-cookies
     A browser window opens; log in, then close it.

Option B – manual cookie export:
  1) Log in to Infineon Developer Center in your browser.
  2) Export cookies (Netscape format) for softwaretools.infineon.com and softwaretools-hosting.infineon.com.
  3) Save the file as: $DEFAULT_COOKIE_FILE
  4) Re-run this script.
EOF
}

resolve_download_url() {
	local original_url="$1"
	local resolved_url="$original_url"
	local resolver_url

	# The hosting frontend /packages/.../download returns an HTML app shell.
	# Use /backend/packages/.../download to obtain a short-lived artifact URL.
	if [[ "$original_url" == https://softwaretools-hosting.infineon.com/packages/*/download ]]; then
		resolver_url="${original_url/https:\/\/softwaretools-hosting.infineon.com\/packages\//https:\/\/softwaretools-hosting.infineon.com\/backend\/packages\/}"
		resolved_url=$(curl -fsS "${CURL_SESSION_ARGS[@]}" "${CURL_AUTH_ARGS[@]}" "$resolver_url" | tr -d '\r\n' || true)
		if [[ "$resolved_url" != https://* ]]; then
			echo "Failed to resolve artifact URL from backend endpoint: $resolver_url" >&2
			return 1
		fi
	fi

	# BEGIN INTERNAL-ONLY (Artifactory) -- remove once all tools ship on the public IDC.
	# Artifactory /ui/native/... is the web UI route; /artifactory/... serves the raw file.
	if [[ "$resolved_url" == https://artifactory.intra.infineon.com/ui/native/* ]]; then
		resolved_url="${resolved_url/\/ui\/native\//\/artifactory\/}"
	fi
	# END INTERNAL-ONLY (Artifactory)

	echo "$resolved_url"
}

url_accepts_session() {
	PYTHONPATH="$SCRIPT_DIR" "$PYTHON" - "$1" <<'PY'
import sys
from security_utils import parse_allowed_url, SESSION_HOSTS

try:
    parse_allowed_url(sys.argv[1], SESSION_HOSTS)
except ValueError:
    raise SystemExit(1)
PY
}

validate_download_url() {
	local url="$1"
	if ! PYTHONPATH="$SCRIPT_DIR" "$PYTHON" - "$url" <<'PY'
import sys
from security_utils import parse_allowed_url

try:
    parse_allowed_url(sys.argv[1])
except ValueError as exc:
    print(f"Rejected artifact URL: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
	then
		return 1
	fi
}

tool_filter=""
platform_filter=""
cookie_file=""
auto_cookies=0
verbose=0
dry_run=0

while [[ $# -gt 0 ]]; do
	case "$1" in
		-h|--help)
			usage
			exit 0
			;;
		--tool)
			tool_filter="${2:-}"
			shift 2
			;;
		--platform)
			platform_filter="${2:-}"
			shift 2
			;;
		--cookie-file)
			cookie_file="${2:-}"
			shift 2
			;;
		--auto-cookies)
			auto_cookies=1
			shift
			;;
		-v|--verbose)
			verbose=1
			shift
			;;
		--dry-run)
			dry_run=1
			shift
			;;
		*)
			echo "Unknown argument: $1" >&2
			usage
			exit 1
			;;
	esac
done

CURL_SESSION_ARGS=()
CURL_AUTH_ARGS=()

# Prefer the model-zoo venv's Python so all dependencies are available.
PYTHON="${SCRIPT_DIR}/../../venv/bin/python3"
[[ -x "$PYTHON" ]] || PYTHON="python3"

if [[ -z "$cookie_file" && -f "$DEFAULT_COOKIE_FILE" ]]; then
	cookie_file="$DEFAULT_COOKIE_FILE"
fi

if [[ -n "$cookie_file" ]]; then
	if [[ ! -f "$cookie_file" ]]; then
		echo "Cookie file not found: $cookie_file" >&2
		print_cookie_help
		exit 1
	fi
	_require_private_file "$cookie_file" "Cookie file" || exit 1
	CURL_SESSION_ARGS=(-b "$cookie_file" -c "$cookie_file")
fi

# Load Bearer token if a previous run already created it.
if [[ -f "$DEFAULT_AUTH_FILE" ]]; then
	_require_private_file "$DEFAULT_AUTH_FILE" "Auth file" || exit 1
	_auth_val=$(cat "$DEFAULT_AUTH_FILE")
	_set_auth_config "$_auth_val"
fi

if [[ -f "$DEFAULT_SESSION_FILE" ]]; then
	_require_private_file "$DEFAULT_SESSION_FILE" "Session file" || exit 1
fi

AUTO_COOKIE_ATTEMPTED=0

try_auto_cookies_if_needed() {
	if [[ $auto_cookies -ne 1 || $AUTO_COOKIE_ATTEMPTED -eq 1 ]]; then
		return 1
	fi

	AUTO_COOKIE_ATTEMPTED=1
	auto_cookie_file="$DEFAULT_COOKIE_FILE"
	echo "Fetching cookies from local browser..."
	if "$PYTHON" "$SCRIPT_DIR/fetch_cookies.py" "$auto_cookie_file"; then
		cookie_file="$auto_cookie_file"
		if [[ ! -f "$cookie_file" ]]; then
			echo "Cookie file not found after auto-cookies: $cookie_file" >&2
			return 1
		fi
		CURL_SESSION_ARGS=(-b "$cookie_file" -c "$cookie_file")

		# Refresh Bearer token after auto-cookies may have written a new auth file.
		CURL_AUTH_ARGS=()
		if [[ -f "$DEFAULT_AUTH_FILE" ]]; then
			_require_private_file "$DEFAULT_AUTH_FILE" "Auth file" || return 1
			_auth_val=$(cat "$DEFAULT_AUTH_FILE")
			_set_auth_config "$_auth_val"
		fi
		return 0
	fi

	echo "Failed to extract cookies automatically. Falling back to manual instructions." >&2
	print_cookie_help
	return 1
}

if [[ ! -f "$MANIFEST" ]]; then
	echo "Manifest not found: $MANIFEST" >&2
	exit 1
fi

if ! "$PYTHON" "$SCRIPT_DIR/validate_manifest.py" "$MANIFEST"; then
	exit 1
fi

_log_verbose "Tool filter: ${tool_filter:-none}, Platform filter: ${platform_filter:-none}"

if [[ $dry_run -eq 1 ]]; then
	echo "[DRY-RUN MODE] No downloads will be performed. Showing what would be downloaded:"
fi

mkdir -p "$OUT_DIR"

downloaded=0
matched=0
failed=0

# `|| [[ -n "$tool_id" ]]` keeps the last row when the manifest lacks a trailing newline.
while IFS=, read -r tool_id version platform filename url auth_required login_required sha256 || [[ -n "$tool_id" ]]; do
	for _f in tool_id version platform filename url auth_required login_required sha256; do
		_v="${!_f}"
		_v="${_v%$'\r'}"
		_v="${_v#"${_v%%[![:space:]]*}"}"
		_v="${_v%"${_v##*[![:space:]]}"}"
		printf -v "$_f" '%s' "$_v"
	done

	if [[ "$tool_id" == "tool_id" || -z "$tool_id" || "$tool_id" == \#* ]]; then
		continue
	fi

	if [[ -n "$tool_filter" && "$tool_id" != "$tool_filter" ]]; then
		continue
	fi

	if [[ -n "$platform_filter" && "$platform" != "$platform_filter" ]]; then
		continue
	fi

	(( ++matched ))
	out_path="$OUT_DIR/$filename"
	headers_path="$OUT_DIR/$filename.headers"
	tmp_path="$OUT_DIR/$filename.part"
	_CURRENT_TMP="$tmp_path"
	_CURRENT_HEADERS="$headers_path"
	resolved_url="$url"

	if [[ -f "$out_path" ]]; then
		if _validate_artifact "$out_path" "$sha256"; then
			echo "Already downloaded and verified: $out_path (skipping)"
			_log_verbose "Validated cached file at $out_path"
			(( ++downloaded ))
			continue
		fi
		echo "Cached artifact is invalid; removing it before download: $out_path" >&2
		rm -f "$out_path"
	fi
	# Backward compatibility: if login_required is not present, fall back to auth_required.
	login_required_norm="${login_required:-$auth_required}"
	login_required_norm="${login_required_norm,,}"

	if [[ "$login_required_norm" == "true" && -z "$cookie_file" ]]; then
		echo "Info: $tool_id is marked login-required. Trying without cookies first (may still work on VPN/internal network)." >&2
	fi

	if [[ $dry_run -eq 1 ]]; then
		echo "[DRY-RUN] Would download: $tool_id ($version, $platform)"
		(( ++downloaded ))
		continue
	fi

	if ! resolved_url=$(resolve_download_url "$url"); then
		# With --auto-cookies, only prompt/login when a login-required URL actually needs resolution.
		if [[ "$login_required_norm" == "true" ]]; then
			if try_auto_cookies_if_needed && resolved_url=$(resolve_download_url "$url"); then
				:
			else
				if [[ $auto_cookies -eq 1 && $AUTO_COOKIE_ATTEMPTED -eq 1 && -z "$cookie_file" ]]; then
					(( ++failed ))
					continue
				fi
			fi
		fi

		if [[ "$resolved_url" == https://* ]]; then
			:
		else
			if [[ "$login_required_norm" == "true" && -f "$DEFAULT_SESSION_FILE" ]]; then
				echo "curl URL resolution failed; trying Playwright download for $tool_id..." >&2
				if "$PYTHON" "$SCRIPT_DIR/playwright_download.py" "$url" "$tmp_path" "$DEFAULT_SESSION_FILE" "$sha256"; then
					if _install_artifact "$tool_id" "$tmp_path" "$out_path"; then
						(( ++downloaded ))
					else
						(( ++failed ))
					fi
				else
					echo "Playwright download failed for $tool_id ($version, $platform)." >&2
					(( ++failed ))
				fi
			else
				[[ "$login_required_norm" == "true" && -z "$cookie_file" ]] && print_cookie_help
				echo "URL resolution failed for $tool_id ($version, $platform)" >&2
				(( ++failed ))
			fi
			continue
		fi
	fi

	if ! validate_download_url "$resolved_url"; then
		(( ++failed ))
		continue
	fi

	echo "Downloading $tool_id ($version, $platform) to $out_path"
	_log_verbose "URL: $resolved_url"

	# Infineon's hosting frontend handles auth through the browser app shell;
	# curl cannot replicate it. Use Playwright directly when a session is available.
	if [[ "$url" == https://softwaretools-hosting.infineon.com/packages/*/download ]]; then
		[[ ! -f "$DEFAULT_SESSION_FILE" ]] && { try_auto_cookies_if_needed || true; }
		if [[ -f "$DEFAULT_SESSION_FILE" ]]; then
			_log_verbose "Using Playwright for hosting URL"
			if "$PYTHON" "$SCRIPT_DIR/playwright_download.py" "$url" "$tmp_path" "$DEFAULT_SESSION_FILE" "$sha256"; then
				if _install_artifact "$tool_id" "$tmp_path" "$out_path"; then
					(( ++downloaded ))
				else
					(( ++failed ))
				fi
			else
				echo "Playwright download failed for $tool_id ($version, $platform)." >&2
				(( ++failed ))
			fi
			continue
		fi
		_log_verbose "No session file for $tool_id; falling back to curl"
	fi

	# Never send the softwaretools cookies/token to unrelated hosts such as Artifactory.
	curl_session=()
	curl_auth=()
	if url_accepts_session "$resolved_url"; then
		curl_session=("${CURL_SESSION_ARGS[@]}")
		curl_auth=("${CURL_AUTH_ARGS[@]}")
	else
		_log_verbose "Host outside SESSION_HOSTS; sending unauthenticated request"
	fi

	if curl -fL --proto '=https' --max-redirs 5 --retry 3 --retry-all-errors "${curl_session[@]}" "${curl_auth[@]}" -D "$headers_path" -o "$tmp_path" "$resolved_url"; then
		:
	else
		curl_status=$?
		echo "Download failed for $tool_id ($version, $platform)" >&2
		_log_verbose "curl failed with exit code $curl_status"
		if [[ -f "$headers_path" ]]; then
			http_code=$(grep -oE '^HTTP/[0-9.]+ [0-9]{3}' "$headers_path" | tail -n1 | grep -oE '[0-9]{3}$' || echo "")
			if [[ "$http_code" == "401" || "$http_code" == "403" ]] && [[ -f "$DEFAULT_SESSION_FILE" ]]; then
				echo "Authentication required; attempting Playwright download for $tool_id..." >&2
				_log_verbose "HTTP $http_code detected, falling back to Playwright"
				rm -f "$tmp_path" "$headers_path"
				if "$PYTHON" "$SCRIPT_DIR/playwright_download.py" "$url" "$tmp_path" "$DEFAULT_SESSION_FILE" "$sha256"; then
					if _install_artifact "$tool_id" "$tmp_path" "$out_path"; then
						(( ++downloaded ))
					else
						(( ++failed ))
					fi
					continue
				fi
				echo "Playwright download also failed for $tool_id." >&2
				_log_verbose "Playwright fallback also failed"
			fi
		fi
		rm -f "$tmp_path" "$headers_path"
		(( ++failed ))
		continue
	fi

	if ! _validate_artifact "$tmp_path" "$sha256"; then
		echo "Rejected artifact for $tool_id ($version, $platform)." >&2
		rm -f "$tmp_path" "$headers_path"
		(( ++failed ))
		continue
	fi

	if _install_artifact "$tool_id" "$tmp_path" "$out_path"; then
		rm -f "$headers_path"
		(( ++downloaded ))
	else
		rm -f "$headers_path"
		(( ++failed ))
	fi
done < "$MANIFEST"

if [[ $matched -eq 0 ]]; then
	echo "No manifest entries matched the provided filters." >&2
	exit 1
fi

if [[ $failed -gt 0 ]]; then
	if [[ $dry_run -eq 1 ]]; then
		echo "[DRY-RUN] Would have downloaded $downloaded tool(s), with $failed error(s)."
	else
		echo "Done with errors. Downloaded $downloaded tool(s), failed $failed tool(s)." >&2
	fi
	exit 1
fi

if [[ $dry_run -eq 1 ]]; then
	echo "[DRY-RUN] Would download $downloaded tool(s) into $OUT_DIR"
else
	echo "Done. Downloaded $downloaded tool(s) into $OUT_DIR"
fi
