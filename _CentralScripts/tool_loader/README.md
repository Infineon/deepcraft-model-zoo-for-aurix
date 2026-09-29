# Model Zoo Tool Loader

A secure, automated tool for downloading Infineon Developer Center tools with intelligent cookie/session management.

## Features

- **Dual Authentication Strategy**
  - Silent mode: Extract cookies from existing browser session (via `browser-cookie3`)
  - Interactive mode: Open browser for login with Playwright
- **Security-First Design**
  - Source URL allowlist validation for every manifest entry
  - HTTPS-only downloads with a bounded redirect count
  - SHA-256 integrity verification when a manifest checksum is provided
  - HTML response and minimum file size validation on every download path
  - Manifest validation for URLs, checksums, filenames, and duplicates
  - Credential files written with owner-only permissions; Bearer token passed via a private curl config, never on the command line
- **Convenient Filtering**
  - Download specific tools by ID: `--tool tsim`
  - Filter by platform: `--platform linux-x64`
  - Combine filters for precision
- **Development Features**
  - Verbose debugging: `--verbose`
  - Dry-run mode: `--dry-run` (preview without downloading)
  - Automatic retry on transient failures
  - Comprehensive error messages

## Model Zoo Setup

The model zoo's one-command setup installs these dependencies and invokes the
loader for every required target tool:

```bash
./_CentralScripts/setup.sh
```

Run that command from the repository root. The instructions below are intended
for troubleshooting or direct downloader development.

## Manual Archive Fallback

If browser authentication works in a regular browser but the automated download
fails, download the three required archives manually and place them in
`_CentralScripts/tool_loader/downloads/`. Follow the repository's
[manual tool download fallback](../../README.md#manual-tool-download-fallback)
for the exact versions, IDC links, filenames, and setup command. This is
different from exporting cookies: setup validates the manually downloaded files
against `tools.csv` and reuses valid archives without authenticating again.

## Standalone Loader Setup

### Prerequisites
- Python 3.9+
- `curl` command-line tool
- Linux or WSL environment

### Setup

```bash
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r _CentralScripts/tool_loader/requirements.txt

# Install Playwright browser binaries (for interactive login)
playwright install chromium

# (Optional) Install system dependencies for Playwright
sudo venv/bin/playwright install-deps chromium
```

## Usage

### Basic Usage

```bash
# Download all tools (requires cookies or interactive login)
./_CentralScripts/tool_loader/tool_loader.sh

# Download with automatic cookie extraction from browser
./_CentralScripts/tool_loader/tool_loader.sh --auto-cookies

# Download a specific tool
./_CentralScripts/tool_loader/tool_loader.sh --tool tsim

# Download for a specific platform
./_CentralScripts/tool_loader/tool_loader.sh --platform linux-x64

# Combine filters
./_CentralScripts/tool_loader/tool_loader.sh --tool tsim --platform linux-x64
```

### Advanced Options

```bash
# Verbose output for debugging network issues
./_CentralScripts/tool_loader/tool_loader.sh --verbose --auto-cookies

# Preview what would be downloaded (no actual downloads)
./_CentralScripts/tool_loader/tool_loader.sh --dry-run

# Use a custom cookie file
./_CentralScripts/tool_loader/tool_loader.sh --cookie-file /path/to/cookies.txt

# Set custom timeout (in seconds, via environment variable)
LOGIN_TIMEOUT=300 ./_CentralScripts/tool_loader/tool_loader.sh --auto-cookies
DOWNLOAD_TIMEOUT=1200 python3 playwright_download.py <url> <output> <session> <sha256>
```

### Full Help

```bash
./_CentralScripts/tool_loader/tool_loader.sh --help
```

## Configuration

### tools.csv

Defines downloadable tools. See [CSV_FORMAT.md](CSV_FORMAT.md) for complete documentation.

### Environment Variables

- `LOGIN_TIMEOUT` (seconds): Timeout for completing Infineon login and receiving an access token. Default: 180s
- `DOWNLOAD_TIMEOUT` (seconds): Timeout for file downloads. Default: 600s (10 min)

## Authentication Methods

### Method 1: Automatic Cookie Extraction (Recommended)

Silently reads cookies from your logged-in browser:

```bash
# First, log in to softwaretools.infineon.com in your browser
# Then run:
./_CentralScripts/tool_loader/tool_loader.sh --auto-cookies
```

**Requires:**
- One of: Chrome, Chromium, Firefox, Edge, Brave, Opera, Vivaldi
- `browser-cookie3` package (installed via `requirements.txt`)

**What happens:**
1. Script reads cookies from your browser's local session
2. If not found, opens a headed Chromium window for interactive login
3. Saves cookies to `infineon.cookies`, auth token to `infineon.auth`, session to `infineon.session.json`

Generated credential files are restricted to the current user (`0600`). Existing
cookie and auth files that are readable by group or others are rejected.

### Method 2: Manual Cookie Export

1. Log in to Infineon Developer Center in your browser
2. Export cookies in Netscape format for:
   - `softwaretools.infineon.com`
   - `softwaretools-hosting.infineon.com`
3. Save as `infineon.cookies` in this directory
4. Run: `./_CentralScripts/tool_loader/tool_loader.sh`

### Method 3: Use Existing Cookies

```bash
./_CentralScripts/tool_loader/tool_loader.sh --cookie-file /path/to/existing/cookies.txt
```

## Security Notes

- ⚠️ **Never commit credential files to git:** `infineon.cookies`, `infineon.auth`, `infineon.session.json`
  - These are automatically ignored (see `.gitignore`)
- All manifest download URLs are validated against the approved Infineon host policy
- Downloads use HTTPS only and follow a bounded number of redirects; the final
  artifact may be served from a pre-signed redirect target (e.g. S3), so integrity
  relies on SHA-256 and credentials are never sent to the redirect target
- SHA-256 checksums are verified when provided
- Downloaded files must pass HTML and minimum-size validation on both curl and Playwright paths
- Bearer tokens are captured separately from cookies and passed to curl via a
  private config file, so they never appear in the process list
- Manifest filenames must be single filenames, not absolute or nested paths

The approved artifact hosts are `softwaretools-hosting.infineon.com`,
`softwaretools.infineon.com`, `softwaretools-preview.icp.infineon.com`, and
`artifactory.intra.infineon.com`. The preview and Artifactory hosts are internal
exceptions and should be removed when no longer required. Credentials are sent
only to the softwaretools and softwaretools-preview hosts, never to Artifactory.

## Troubleshooting

### "Cookie file not found"
```bash
# Run authentication setup
./_CentralScripts/tool_loader/tool_loader.sh --auto-cookies
```

### "Login timed out"
- The Infineon SSO took longer than expected
- Increase timeout: `LOGIN_TIMEOUT=300 ./_CentralScripts/tool_loader/tool_loader.sh --auto-cookies`
- Check your internet connection
- A temporary `oidc.<state>` browser entry is not a completed login. The loader
  waits for an `oidc.user:*` entry containing a Bearer access token and reports
  the final page if no token is received.

### "Download timed out"
- The file is large or connection is slow
- Increase timeout: `DOWNLOAD_TIMEOUT=1200 python3 playwright_download.py ...`
- Try again (script includes automatic retries)

### "Rejected artifact URL from untrusted host"
- URL is from an unexpected domain
- Verify the URL is correct in `tools.csv`
- If legitimate, update the approved host policy in `security_utils.py`

### "Validation failed: response is HTML"
- Login/authentication failed
- Run `./_CentralScripts/tool_loader/tool_loader.sh --auto-cookies` to refresh credentials
- Check that you're logged into Infineon Developer Center in your browser

### "SHA-256 mismatch"
- Downloaded file is corrupted
- Check your internet connection and try again
- Verify the `sha256` value in `tools.csv`

### Verbose debugging
```bash
./_CentralScripts/tool_loader/tool_loader.sh --verbose --auto-cookies 2>&1 | tee debug.log
```

## Project Structure

```
.
├── tool_loader.sh              # Main orchestration script
├── fetch_cookies.py            # Cookie/session extraction (browser-cookie3 + Playwright)
├── playwright_download.py      # Authenticated file download with retry logic
├── security_utils.py           # Shared URL, artifact, filename, and secret-file checks
├── validate_manifest.py        # Fail-fast tools.csv validation
├── tools.csv                   # Tool manifest (add tools here)
├── requirements.txt            # Python dependencies (pinned versions)
├── README.md                   # This file
└── CSV_FORMAT.md               # Detailed tools.csv documentation
```

## Development

### Adding Type Hints

Python files include full type hints for IDE support and type checking:

```bash
# Check types with mypy
pip install mypy
mypy fetch_cookies.py playwright_download.py
```

### Testing

```bash
# Test with dry-run first
./_CentralScripts/tool_loader/tool_loader.sh --dry-run

# Test specific tool
./_CentralScripts/tool_loader/tool_loader.sh --tool tsim --dry-run --verbose

# Full integration test (requires auth)
./_CentralScripts/tool_loader/tool_loader.sh --tool tsim --auto-cookies --verbose

# Run security and manifest regression tests
python3 -m unittest discover -s tests -v
```

## License

Internal Infineon tool. Licensing is governed by the repository owner.

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Run with `--verbose` flag to enable debug output
3. Contact the Model Zoo team
