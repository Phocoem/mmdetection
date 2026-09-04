# BS-DGCFPN Mask R-CNN config.
_base_ = './mask_rcnn_r50_fpn.py'

custom_imports = dict(
    imports=['mmdet.models.necks.bs_dgcfpn'],
    allow_failed_imports=False
)

model = dict(
    neck=dict(
        type='BSDGCFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        context_dilations=(1, 3, 5),
        context_branch_channels=64,
        context_with_image_pool=False,
        use_context=True,
        use_detail=True,
        use_adaptive_gate=True,
        use_foreground_gate=False,
        static_weights=(0.7, 0.15, 0.15),
        gate_reduction=4,
        gate_init_bias=(2.0, -1.0, -1.0),
        fg_hidden_channels=64,
        fg_init_bias=-0.5,
        residual_alpha=0.3,
        apply_to_levels=(0, 1, 2),

    )
)

work_dir = 'work_dirs/research/mask_rcnn_r50_dgcfpn_no_fg/seed_2025_3'
