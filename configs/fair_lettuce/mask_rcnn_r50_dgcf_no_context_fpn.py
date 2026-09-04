# Ablation: DGCFPN without ASPP-like context branch.
# Tests whether context is necessary beyond detail/adaptive fusion.
_base_ = './mask_rcnn_r50_fpn.py'

custom_imports = dict(
    imports=['mmdet.models.necks.dgcf_fpn'],
    allow_failed_imports=False)

model = dict(
    neck=dict(
        type='DGCFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        use_context=False,
        use_detail=True,
        use_adaptive_gate=True,
        gate_reduction=4,
        gate_init_bias=(1.0, 0.0, 0.0),
        residual_alpha=1.0,
        apply_to_levels=(0, 1, 2, 3, 4),
    )
)

work_dir = 'work_dirs/research/mask_rcnn_r50_dgcf_no_context_fpn/seed_2025_3'
