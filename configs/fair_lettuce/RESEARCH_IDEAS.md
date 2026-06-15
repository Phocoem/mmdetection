# Research Direction: Robust Frequency-Adaptive Pyramid

## Main idea

Extend `FPFPN` into a Frequency-Adaptive FPN (FA-FPFPN). Instead of two global
scalars for all pyramid levels, predict bounded high-frequency and
low-frequency fusion gates per level and per image:

`P_l = L_l + g_low_l * Low(U(P_l+1)) + g_high_l * High(U(P_l+1))`

The gates are produced from compact global statistics of the lateral feature.
This lets the model suppress unreliable high-frequency detail under noise while
preserving boundaries on clean or low-contrast images.

## Novel training signal

Use paired clean/corrupted views only during training and add a frequency
consistency loss:

- Low-frequency mask logits should remain stable under brightness changes.
- High-frequency boundary logits should remain stable under moderate noise.
- The clean-view detector remains the teacher through stop-gradient, so no
  additional annotation is required.

## Falsifiable hypotheses

1. Per-level gates improve segmentation AP over global `alpha` and `beta`.
2. Frequency consistency improves robustness AP on brightness/noise test sets
   without reducing clean AP.
3. Early pyramid levels learn larger high-frequency gates; deep levels learn
   larger low-frequency gates.

## Required ablations

- FPN vs current FPFPN vs FA-FPFPN.
- Global gates vs per-level gates vs per-image per-level gates.
- No consistency loss vs low-only vs high-only vs both.
- Clean AP, brightness AP, noise AP, boundary F-score, parameters, FLOPs, and
  latency.
- At least three seeds using the shared fair protocol.

## Strong reporting choice

Report both accuracy and robustness drop:

`Robustness drop = clean segm_mAP - corrupted segm_mAP`

This separates a model that is simply stronger everywhere from one that is
specifically more stable under domain corruption.
