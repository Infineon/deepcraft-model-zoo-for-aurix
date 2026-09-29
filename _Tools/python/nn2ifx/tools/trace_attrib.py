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
"""Shared instruction-trace attributor for the trace-based profilers.

Both the TriCore TSIM profiler ([tsim_profile.py]) and the PPU nSIM CPI
estimator ([arc_cpi.py]) walk a linear instruction trace once, isolate the
inference region (from the first time the PC equals ``entry()``'s start until
control first re-enters ``main()``), and attribute every in-region instruction
to the ONNX ``node_*`` whose function is currently executing — folding nested
runtime-helper calls into that node (inclusive basis) and collecting ``entry()``
glue under ``"__entry__"``.

The two simulators differ only in their per-line text format and in what each
instruction carries: TSIM supplies a per-instruction cycle cost (the delta of a
cumulative cycle counter), while nSIM supplies only the instruction mnemonic
(for a downstream CPI histogram). Both aspects are captured here by a
caller-supplied ``line_parser`` returning ``(pc, cost, mnemonic)``.
"""

from typing import Callable, Optional, Tuple

# A parsed trace record: (program_counter, per-instruction cost or None,
# mnemonic or None). ``line_parser`` returns this, or None to skip a line.
TraceRecord = Tuple[int, Optional[int], Optional[str]]
LineParser = Callable[[str], Optional[TraceRecord]]


class TraceAttributor:
    """Single-pass attribution of a linear instruction trace to nodes + totals.

    Feed raw trace lines one at a time via :meth:`feed`. When the inference
    region ends (control re-enters ``main()``) :attr:`done` becomes True; a
    caller reading a huge trace from a file may stop as soon as it is set, while
    a caller draining a live pipe should keep feeding (further lines are
    ignored) so the producer does not block.

    Accumulated results (all keyed by node name, with ``entry()`` glue under
    ``"__entry__"``):

    * :attr:`region_instructions` / :attr:`region_cost` — region totals.
    * :attr:`node_instructions` / :attr:`node_cost` — per-node totals.
    * :attr:`region_mnemonics` — ``{mnemonic: count}`` region histogram.
    * :attr:`node_mnemonics` — ``{node: {mnemonic: count}}`` per-node histograms.

    ``cost`` sums are only populated when the parser supplies a cost; mnemonic
    histograms only when it supplies a mnemonic.
    """

    def __init__(
        self,
        entry_lo: int,
        entry_hi: int,
        main_lo: int,
        main_hi: int,
        node_starts: dict,
        line_parser: LineParser,
    ):
        self.entry_lo, self.entry_hi = entry_lo, entry_hi
        self.main_lo, self.main_hi = main_lo, main_hi
        self.node_starts = node_starts
        self._parse = line_parser

        self.region_instructions = 0
        self.region_cost = 0
        self.region_mnemonics: dict = {}
        self.node_instructions: dict = {}
        self.node_cost: dict = {}
        self.node_mnemonics: dict = {}

        self._current = None
        self._in_region = False
        self._started = False
        self._done = False

    def feed(self, line: str) -> None:
        if self._done:
            return
        rec = self._parse(line)
        if rec is None:
            return
        pc, cost, mnemonic = rec

        if not self._in_region:
            if pc == self.entry_lo:
                self._in_region = True
                self._started = True
            else:
                return
        elif self.main_lo <= pc < self.main_hi:
            self._done = True
            return

        # Attribute to the current node (inclusive of nested helpers).
        name = self.node_starts.get(pc)
        if name is not None:
            self._current = name
        elif self.entry_lo <= pc < self.entry_hi:
            self._current = "__entry__"
        cur = self._current

        self.region_instructions += 1
        self.node_instructions[cur] = self.node_instructions.get(cur, 0) + 1
        if cost is not None:
            self.region_cost += cost
            self.node_cost[cur] = self.node_cost.get(cur, 0) + cost
        if mnemonic is not None:
            self.region_mnemonics[mnemonic] = self.region_mnemonics.get(mnemonic, 0) + 1
            hist = self.node_mnemonics.setdefault(cur, {})
            hist[mnemonic] = hist.get(mnemonic, 0) + 1

    @property
    def done(self) -> bool:
        """True once control has re-entered ``main()`` (region complete)."""
        return self._done

    @property
    def region_found(self) -> bool:
        """True if ``entry()`` was reached (i.e. a region was isolated)."""
        return self._started
