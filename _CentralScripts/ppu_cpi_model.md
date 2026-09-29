# PPU cycle estimation (ARC CPI model)

When a model is converted for the **`tc4dx_ppu`** target (AURIX™ TC4x Parallel
Processing Unit, an ARC VPX vector DSP), the conversion service reports the PPU
performance as an **estimated** cycle count (method label *"nSIM + ARC CPI"* in
`results.json` and the benchmark tables). This document explains how that
estimate is produced and how accurate it is against real PPU hardware.

## Why the PPU result is estimated

The bundled ARC nSIM (FREE flavour) is **functional-only** — it executes the
generated code and validates numerics, but it has no cycle counter. The PPU
cycle count is therefore reconstructed from the instruction stream rather than
measured by the simulator.

## How cycles are estimated

nSIM produces an instruction histogram (how many of each ARC VPX instruction the
inference executes). Each instruction class is assigned a cost (cycles per
instruction) and the estimate is the weighted sum:

```
cycles ≈ Σ  count(class) × CPI(class)
```

The class costs live in
[`arc_cpi.py`](../_Tools/python/nn2ifx/tools/arc_cpi.py) (`CLASS_CPI`). Vector
lane width is already captured by the histogram: an FP32 (8-lane) MAC emits ~4×
more vector instructions than the INT8 (32-lane) path, so no per-lane scaling is
applied.

## Memory hierarchy (`memconfig 101`)

A pure instruction sum assumes every operand is equally fast to reach, which is
not true on the PPU. onnx2c-ifx places tensors across three tiers (default
`--memconfig 101`), filling the fastest first and spilling the rest:

| Tier | Size | Holds | Load latency |
|---|---|---|---|
| VCCM | 116 KB | resident tensors (`__vccm`) | ~2 cyc |
| CSM_RW | 512 KB | spilled **activations** (variables) | ~2 cyc |
| Flash | 4 MB | spilled **weights** (constants) | ~45 cyc |

Which tensors spill, and to which tier, is read from the onnx2c `model.md`
report generated alongside `model.c`.

## Memory-hierarchy corrections

Because reaching a spilled operand costs far more than the flat load assumed by
the instruction sum, two additive corrections are applied to the PPU estimate
(only for `tc4dx_ppu`, and only when a model actually spills):

- **Spilled-weight Flash load** — a low-reuse layer whose weights spill to Flash
  (e.g. large dense/MLP matmuls) pays the ~45-cyc Flash latency per weight
  vector-load instead of ~2 cyc. Compute-bound convolutions, whose weight loads
  are hidden behind arithmetic, are not charged.
- **Spilled-activation CSM streaming** — layers that stream a large volume of
  activations through CSM_RW (typically FP32 CNNs, whose 4-byte activations
  produce the most traffic) pay a per-byte streaming cost.

Both terms are derived per model from `model.md` + the ONNX graph in
[`ppu_residency.py`](../_Tools/python/nn2ifx/tools/ppu_residency.py) and applied
by [`arc_cpi.py`](../_Tools/python/nn2ifx/tools/arc_cpi.py); models that fit
entirely in VCCM are unaffected.

## Accuracy

Across the model zoo the estimate tracks measured PPU hardware to a **mean
HW/estimate ratio of ≈ 1.34×** (i.e. the estimate is a modest, consistent
underestimate rather than a random one). Representative points:

| Model | Workload | HW / estimate |
|---|---|---|
| cnn_weather (FP32) | FP32 CNN, activation-streaming | ~1.00× |
| RUL_MLP | FP32 MLP, weights in Flash | ~0.98× |
| MNIST_MLP | FP32 MLP, weights in Flash | ~1.01× |
| MNIST_CNN | FP32 CNN, fully resident | ~1.27× |
| cnn_weather (INT8/INT16) | quantized CNN | ~1.2–1.4× |
| RUL_LSTM / vehicle_tracker_lstm | recurrent | ~1.3–1.5× |

## Limitations

The PPU figure is an **estimate, not a cycle-accurate measurement**. Known
residuals:

- **Very small models** carry a fixed first-run instruction-fetch (cold-start)
  cost that dominates their tiny cycle budget and is not part of the steady-state
  model.
- **Recurrent models** (LSTM/GRU) are underestimated because a throughput sum
  does not model the latency of serial dependency chains across timesteps.

Use the PPU cycle count for relative comparison and order-of-magnitude sizing;
for a cycle-accurate figure, measure on target.
