# Ablation: DGCFPN without detail branch.
# Tests whether the detail/high-frequency path helps blur/motion robustness.
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
        context_dilations=(1, 3, 6, 9),
        context_branch_channels=64,
        context_with_image_pool=True,
        use_context=True,
        use_detail=False,
        use_adaptive_gate=True,
        gate_reduction=4,
        gate_init_bias=(1.0, 0.0, 0.0),
        residual_alpha=1.0,
        apply_to_levels=(0, 1, 2, 3, 4),
    )
)

work_dir = 'work_dirs/research/mask_rcnn_r50_dgcf_no_detail_fpn/seed_2026_3'
