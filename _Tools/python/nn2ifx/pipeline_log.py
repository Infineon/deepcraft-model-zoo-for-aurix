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
from pathlib import Path
from datetime import datetime


class PipelineLog:
    """Structured plain-text log for the nn2ifx pipeline.

    Writes a human-readable log file organized by pipeline steps,
    capturing commands, tool output, and results.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w")
        self._write_header()

    def _write_header(self):
        self._file.write(f"{'='*72}\n")
        self._file.write("nn2ifx Pipeline Log\n")
        self._file.write(f"Started: {datetime.now().isoformat()}\n")
        self._file.write(f"{'='*72}\n\n")

    def begin_step(self, name: str):
        """Mark the beginning of a pipeline step."""
        self._file.write(f"\n{'─'*72}\n")
        self._file.write(f"[{datetime.now().strftime('%H:%M:%S')}] {name}\n")
        self._file.write(f"{'─'*72}\n\n")
        self._file.flush()

    def log_command(self, cmd: list):
        """Log a shell command."""
        self._file.write(f"Command:\n  {' '.join(cmd)}\n\n")
        self._file.flush()

    def log_output(self, output: str, label: str = "Output"):
        """Log tool output (stdout/stderr)."""
        if output and output.strip():
            self._file.write(f"{label}:\n")
            for line in output.strip().splitlines():
                self._file.write(f"  {line}\n")
            self._file.write("\n")
            self._file.flush()

    def log_info(self, msg: str):
        """Log an informational message."""
        self._file.write(f"{msg}\n")
        self._file.flush()

    def log_error(self, msg: str):
        """Log an error message."""
        self._file.write(f"ERROR: {msg}\n")
        self._file.flush()

    def log_result(self, key: str, value: str):
        """Log a key-value result."""
        self._file.write(f"  {key}: {value}\n")
        self._file.flush()

    def close(self):
        """Finalize and close the log file."""
        self._file.write(f"\n{'='*72}\n")
        self._file.write(f"Finished: {datetime.now().isoformat()}\n")
        self._file.write(f"{'='*72}\n")
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        try:
            self.close()
        except (IOError, OSError):
            pass
