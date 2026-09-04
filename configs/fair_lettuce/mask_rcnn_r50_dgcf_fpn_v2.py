# DGCFPNv2 — mô hình đề xuất sau cải tiến dựa trên phân tích 3-seed:
# 1) zero_init_branches=True: output ban đầu = FPN chuẩn 100%, giảm seed variance
# 2) gate_init_bias=(4,-2,-2): softmax ~ [0.98, 0.01, 0.01]
# 3) detail_light=True: detail branch 1x1 conv đúng như mô tả paper
# 4) record_gate_stats bật khi evaluate để phân tích "adaptive" (MC3)
_base_ = './mask_rcnn_r50_fpn.py'

custom_imports = dict(
    imports=['mmdet.models.necks.dgcf_fpn',
             'mmdet.models.necks.dgcf_fpn_v2'],
    allow_failed_imports=False)

model = dict(
    neck=dict(
        type='DGCFPNv2',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        context_dilations=(1, 3, 6, 9),
        context_branch_channels=64,
        context_with_image_pool=True,
        use_context=True,
        use_detail=True,
        detail_light=True,
        detail_avg_kernel=3,
        use_adaptive_gate=True,
        gate_reduction=4,
        gate_init_bias=(4.0, -2.0, -2.0),
        zero_init_branches=True,
        record_gate_stats=False,  # bật True khi chạy dump_gate_weights.py
        residual_alpha=1.0,
        apply_to_levels=(0, 1, 2, 3, 4),
    ))

work_dir = 'work_dirs/research/mask_rcnn_r50_dgcf_fpn_v2/default'
