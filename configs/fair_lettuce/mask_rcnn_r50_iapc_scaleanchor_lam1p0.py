# IAPC CAI TIEN (scale-anchor) - lambda=1.0, uu tien test truoc vi day la
# noi lo hong magnitude the hien ro nhat (tang 99x o P5, APcorr te nhat
# trong ca sweep - 0.692, thap hon ca control 0.699).
#
# CHI KHAC config goc (mask_rcnn_r50_iapc.py) DUNG 2 CHO: model.type va
# them model.scale_anchor_cap - moi thu khac (consistency_weight=1.0,
# corruption_cfg, ...) GIU NGUYEN de phep so sanh "A" (vs control) va
# "B" (vs ban goc cung lambda) chi khac dung 1 bien (co/khong scale-anchor).
_base_ = './mask_rcnn_r50_iapc.py'
custom_imports = dict(
    imports=['mmdet.models.detectors.consistency_mask_rcnn_scale_anchor'],
    allow_failed_imports=False)
model = dict(
    type='IAPCMaskRCNNScaleAnchor',
    # Tran tren cho scale s_l(x), ap dung CHUNG cho ca 4 tang P2-P5.
    # Co so chon 1.0: baseline (khong consistency) co magnitude tu nhien
    # ~0.31-0.36 o moi tang (do bang plot_feature_heatmaps.py) - dat tran
    # o ~3x muc do de mang van co du khong gian hoc, nhung chan dut kha
    # nang phinh 30-170x da quan sat duoc o ban goc.
    scale_anchor_cap=1.0,
)
work_dir = 'work_dirs/research/mask_rcnn_r50_iapc_scaleanchor_lam1p0'
