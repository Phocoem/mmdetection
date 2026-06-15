# Fair Lettuce Instance Segmentation

All experiment configs are in this directory. They share one protocol from
`_base_/fair_protocol.py`; experiment files change only the architecture and
`work_dir`.

## Controlled protocol

- Dataset: the same train, validation, and test annotations/splits.
- Input: resize to 800 x 800 with preserved aspect ratio; random horizontal
  flip is the only training augmentation.
- Budget: batch size 2, validation every epoch, with a 200-epoch safety cap.
- Convergence: reduce learning rate on validation plateau and stop after 20
  validation epochs without at least 0.001 absolute segmentation mAP gain.
- Optimization: AdamW, learning rate 1e-4 and weight decay 0.05.
- R50 backbone training: identical ImageNet initialization, frozen stage, and
  BatchNorm behavior.
- Evaluation: COCO segmentation mAP on the same split.
- Reproducibility: seed 2026, deterministic mode, no experiment checkpoint
  loaded or resumed.
- Logging: text, JSON scalars, TensorBoard, resolved config, best checkpoint,
  and last checkpoint.

Run the fairness audit before training:

```powershell
python configs/fair_lettuce/audit_fairness.py
```

Use `--resolved` in a healthy MMEngine environment to compare the fully merged
configs as well.

Train one experiment:

```powershell
python tools/train.py configs/fair_lettuce/mask_rcnn_r50_fpn.py
```

For the complete train, clean-test, corruption-test, and reporting workflow,
use [tools/research/README.md](../../tools/research/README.md). The
publication-oriented protocol is documented in
[RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md).

Architecture-specific pretrained backbone initialization inherited from each
official base config remains part of the architecture. For the strictest
from-scratch comparison, disable every backbone `init_cfg` in a separate
ablation.
