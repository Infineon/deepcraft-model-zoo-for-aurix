# tools.csv Format Documentation

The `tools.csv` file defines which tools can be downloaded. Each row represents a downloadable tool artifact.

## Column Definitions

| Column | Type | Description |
|--------|------|-------------|
| `tool_id` | string | Unique identifier for the tool (e.g., `tsim`, `aurixgcc`). Used with `--tool` filter. |
| `version` | string | Tool version (e.g., `1.18.196`, `03-2026`). For filtering and documentation. |
| `platform` | string | Target platform (e.g., `linux-x64`, `windows-x86`). Used with `--platform` filter. |
| `filename` | string | Output filename in the `downloads/` directory. Should be unique. |
| `url` | string | Direct download URL from Infineon. Must use one of the allowed domains (see below). |
| `auth_required` | boolean | **Deprecated**. Whether authentication is required. Use `login_required` instead. |
| `login_required` | boolean | Whether user login to Infineon Developer Center is required (`true`/`false`). |
| `sha256` | string | Optional SHA-256 checksum for integrity verification. If empty, no checksum validation occurs. |

## URL Requirements

All download URLs must use HTTPS and come from one of these approved domains:
- `softwaretools-hosting.infineon.com`
- `softwaretools.infineon.com`
- `softwaretools-preview.icp.infineon.com` (internal exception)
- `artifactory.intra.infineon.com` (internal exception)

URLs from other domains, URLs containing user information, and non-default
ports are rejected for security reasons. Credentials are never sent to
Artifactory.

## Filename and Checksum Requirements

`filename` must be a single ordinary filename. Absolute paths, path separators,
`..`, and duplicate filenames are rejected before any network access. `tool_id`
values must also be unique.

`sha256` may be empty for legacy entries, but a 64-character hexadecimal SHA-256
checksum is strongly recommended. When present, it is verified for both curl
and Playwright downloads. All download paths also reject HTML responses and
artifacts smaller than 10 KiB.

## Examples

```csv
tool_id,version,platform,filename,url,auth_required,login_required,sha256
tsim,1.18.196,linux-x64,tsim_1.18.196_linux_x64.deb,https://softwaretools-hosting.infineon.com/packages/com.ifx.tb.tool.tsimtricoreinstructionsetsimulator/versions/1.18.196/artifacts/tsim_1.18.196_linux_x64.deb/download,true,true,abc123def456...
aurixgcc,03-2026,linux-x64,aurixgcc_03-2026_Linux_x86-x64.zip,https://softwaretools-hosting.infineon.com/packages/com.ifx.tb.tool.aurixgcc/versions/03-2026/artifacts/aurixgcc_03-2026_Linux_x86-x64.zip/download,false,false,
```

## Adding New Tools

1. Obtain the download URL from Infineon Developer Center
2. Compute SHA-256 checksum (recommended for security):
   ```bash
   sha256sum <artifact>
   ```
3. Add a row to `tools.csv` with the correct metadata
4. Test with `--dry-run` first:
   ```bash
   ./_CentralScripts/tool_loader/tool_loader.sh --dry-run
   ```
