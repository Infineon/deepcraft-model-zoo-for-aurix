# Copyright (c) 2025, Infineon Technologies AG, or an affiliate of Infineon Technologies AG. All rights reserved.
#
# This software, associated documentation and materials ("Software") is owned by Infineon Technologies AG or one of
# its affiliates ("Infineon") and is protected by and subject to worldwide patent protection, worldwide copyright laws,
# and international treaty provisions. Therefore, you may use this Software only as provided in the license agreement
# accompanying the software package from which you obtained this Software. If no license agreement applies, then any use,
# reproduction, modification, translation, or compilation of this Software is prohibited without the express written
# permission of Infineon.
#
# Disclaimer: UNLESS OTHERWISE EXPRESSLY AGREED WITH INFINEON, THIS SOFTWARE IS PROVIDED AS-IS, WITH NO WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, ALL WARRANTIES OF NON-INFRINGEMENT OF THIRD-PARTY RIGHTS AND IMPLIED
# WARRANTIES SUCH AS WARRANTIES OF FITNESS FOR A SPECIFIC USE/PURPOSE OR MERCHANTABILITY. Infineon reserves the right to make
# changes to the Software without notice. You are responsible for properly designing, programming, and testing the
# functionality and safety of your intended application of the Software, as well as complying with any legal requirements
# related to its use. Infineon does not guarantee that the Software will be free from intrusion, data theft or loss, or other
# breaches ("Security Breaches"), and Infineon shall have no liability arising out of any Security Breaches. Unless otherwise
# explicitly approved by Infineon, the Software may not be used in any application where a failure of the Product or any
# consequences of the use thereof can reasonably be expected to result in personal injury.

"""Thin wrapper around the nn2ifx benchmarking pipeline.

Exposes a small, stable surface used by the Flask conversion service:

* :data:`TARGETS`            — the supported hardware targets (canonical names)
* :func:`resolve_target`     — map a user-supplied name (incl. legacy ``TC3``/
                               ``TC4``) to a canonical target
* :func:`resolve_targets`    — expand a target spec (single, list, ``all``/
                               ``compare``) into a list of canonical targets
* :class:`ModelConverter`    — convert + benchmark a single ONNX model for one
                               target, or compare it across several targets
"""

from pathlib import Path
from typing import Iterable, List, Optional, Union

from nn2ifx.devices import TARGETS as _DEVICE_TARGETS
from pipeline import run_target, run_compare

# Canonical targets supported by the pipeline.
TARGETS: List[str] = list(_DEVICE_TARGETS.keys())  # tc4dx, tc3xx, tc4dx_ppu, arm_m4

# Backwards-compatible aliases for the names used by older notebooks/clients.
TARGET_ALIASES = {
    "tc3": "tc3xx",
    "tc4": "tc4dx",
    "tc3x": "tc3xx",
    "tc4x": "tc4dx",
    "tc3xx": "tc3xx",
    "tc4dx": "tc4dx",
    "tc4dx_ppu": "tc4dx_ppu",
    "ppu": "tc4dx_ppu",
    "arm_m4": "arm_m4",
    "armm4": "arm_m4",
}

# Spec values that request a cross-platform comparison over *all* targets.
COMPARE_KEYWORDS = {"compare", "all"}


def resolve_target(name: str) -> str:
    """Map a user-supplied target name to a canonical pipeline target."""
    key = TARGET_ALIASES.get(str(name).strip().lower())
    if key is None or key not in _DEVICE_TARGETS:
        raise ValueError(
            f"Invalid target '{name}'. Must be one of "
            f"{TARGETS} (aliases: TC3, TC4)."
        )
    return key


def resolve_targets(spec: Union[str, Iterable[str]]) -> List[str]:
    """Expand a target spec into a list of canonical targets.

    Accepts a single name, an iterable of names, a comma/space separated
    string, or the keywords ``compare``/``all`` (which expand to every target).
    """
    if isinstance(spec, str):
        tokens = [t for t in spec.replace(",", " ").split() if t]
    else:
        tokens = [str(t) for t in spec]

    if len(tokens) == 1 and tokens[0].strip().lower() in COMPARE_KEYWORDS:
        return list(TARGETS)

    resolved: List[str] = []
    for tok in tokens:
        canonical = resolve_target(tok)
        if canonical not in resolved:
            resolved.append(canonical)
    if not resolved:
        raise ValueError("No valid targets specified.")
    return resolved


def is_compare_spec(spec: Union[str, Iterable[str]]) -> bool:
    """Return True when the spec requests a multi-target comparison."""
    if isinstance(spec, str):
        tokens = [t for t in spec.replace(",", " ").split() if t]
    else:
        tokens = [str(t) for t in spec]
    if len(tokens) == 1 and tokens[0].strip().lower() in COMPARE_KEYWORDS:
        return True
    return len(tokens) > 1


class ModelConverter:
    """Convert and benchmark a single ONNX model via the nn2ifx pipeline."""

    def __init__(
        self,
        model: Union[str, Path],
        test_data: Union[str, Path],
        out_dir: Union[str, Path],
        profile: bool = False,
        profiler: str = "tsim",
    ):
        self.model = Path(model)
        assert self.model.exists(), f"model {self.model} does not exist"
        self.test_data = Path(test_data)
        assert self.test_data.exists(), f"test data dir {self.test_data} does not exist"
        self.out_dir = Path(out_dir)
        self.profile = profile
        self.profiler = profiler

    def run(self, target: str) -> dict:
        """Run the pipeline for one target. Returns the harmonized result dict.

        Output files are written to ``<out_dir>/<target>/`` and include
        ``model.c``, ``main.c``, ``model.elf``, ``model.md``, ``pipeline.log``,
        ``results.json`` and (on TriCore TSIM runs) ``model.tsim_prof.log``.
        """
        canonical = resolve_target(target)
        target_out = self.out_dir / canonical
        return run_target(
            target=canonical,
            model=self.model,
            test_data=self.test_data,
            out=target_out,
            profile=self.profile,
            profiler=self.profiler,
        )

    def compare(self, targets: Optional[Iterable[str]] = None) -> dict:
        """Benchmark the model across several targets and aggregate a table.

        Writes ``comparison.txt``/``comparison.json`` (plus per-target
        ``results.json``) under ``<out_dir>/``. Returns the aggregated dict.
        """
        target_list = (
            list(TARGETS) if targets is None else [resolve_target(t) for t in targets]
        )
        return run_compare(
            targets=target_list,
            model=self.model,
            test_data=self.test_data,
            out=self.out_dir,
            profile=self.profile,
            profiler=self.profiler,
        )
