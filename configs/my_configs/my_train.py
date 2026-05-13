import os
import sys
import subprocess

os.makedirs("configs/my_models", exist_ok=True)

data_root = "data/mmdet_dataset/"
classes = ("lettuce",)
num_classes = 1

common = f"""
data_root = '{data_root}'

metainfo = {{
    'classes': {classes},
    'palette': [(220, 20, 60)]
}}

num_classes = {num_classes}

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/')
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/')
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'annotations/val.json',
    metric=['bbox', 'segm']
)

test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_interval=5
)

default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=3),
    logger=dict(interval=50)
)
"""

configs = {
    "mask_r50": {
        "filename": "mask_rcnn_r50_lettuce.py",
        "content": f"""
_base_ = '../mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'

{common}

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
        mask_head=dict(num_classes=num_classes)
    )
)

load_from = 'checkpoints/mask_rcnn_r50.pth'
work_dir = './work_dirs/mask_rcnn_r50_lettuce'
"""
    },

    "mask_r101": {
        "filename": "mask_rcnn_r101_lettuce.py",
        "content": f"""
_base_ = '../mask_rcnn/mask-rcnn_r101_fpn_1x_coco.py'

{common}

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
        mask_head=dict(num_classes=num_classes)
    )
)

load_from = 'checkpoints/mask_rcnn_r101.pth'

work_dir = './work_dirs/mask_rcnn_r101_lettuce'
"""
    },

    "yolact": {
        "filename": "yolact_r50_lettuce.py",
        "content": f"""
_base_ = '../yolact/yolact_r50_1xb8-55e_coco.py'

{common}

model = dict(
    bbox_head=dict(num_classes=num_classes)
)

load_from = 'checkpoints/yolact_r50.pth'

work_dir = './work_dirs/yolact_r50_lettuce'
"""
    },

    "solo": {
        "filename": "solo_r50_lettuce.py",
        "content": f"""
_base_ = '../solo/solo_r50_fpn_1x_coco.py'

{common}

model = dict(
    mask_head=dict(num_classes=num_classes)
)

load_from = 'checkpoints/solo_r50.pth'

work_dir = './work_dirs/solo_r50_lettuce'
"""
    },

    "solov2": {
        "filename": "solov2_r50_lettuce.py",
        "content": f"""
_base_ = '../solov2/solov2_r50_fpn_1x_coco.py'

{common}

model = dict(
    mask_head=dict(num_classes=num_classes)
)

load_from = 'checkpoints/solov2_r50.pth'
work_dir = './work_dirs/solov2_r50_lettuce'
"""
    },

"condinst": {
    "filename": "condinst_r50_lettuce.py",
    "content": f"""
_base_ = '../condinst/condinst_r50_fpn_ms-poly-90k_coco_instance.py'

{common}

model = dict(
    bbox_head=dict(
        num_classes=num_classes
    )
)

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=10000,
    val_interval=1000
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=1000, max_keep_ckpts=3),
    logger=dict(interval=50)
)

load_from = 'checkpoints/condinst_r50.pth'
work_dir = './work_dirs/condinst_r50_lettuce'
"""
}
}


def create_config_files():
    for model_name, cfg in configs.items():
        path = os.path.join("configs/my_models", cfg["filename"])

        with open(path, "w", encoding="utf-8") as f:
            f.write(cfg["content"].strip() + "\n")

        print(f"Created: {path}")


def train_model(model_name):
    if model_name not in configs:
        print("Model không tồn tại!")
        print("Các model có thể train:")
        for name in configs:
            print("-", name)
        return

    cfg_path = os.path.join("configs/my_models", configs[model_name]["filename"])

    print("=" * 80)
    print("Training model:", model_name)
    print("Config:", cfg_path)
    print("=" * 80)

    subprocess.run(
        [sys.executable, "tools/train.py", cfg_path],
        check=True
    )


if __name__ == "__main__":
    create_config_files()

    if len(sys.argv) < 2:
        print("\nBạn chưa chọn model.")
        print("Cách dùng:")
        print("python create_configs_and_train.py mask_r50")
        print("python create_configs_and_train.py mask_r101")
        print("python create_configs_and_train.py yolact")
        print("python create_configs_and_train.py solo")

        print("\nCác model có sẵn:")
        for name in configs:
            print("-", name)

        sys.exit()

    selected_model = sys.argv[1]
    train_model(selected_model)