import json
import cv2
import random
import shutil
from pathlib import Path


DATASET_DIR = Path("dataset")
IMAGE_DIR = DATASET_DIR / "images"
LABEL_DIR = DATASET_DIR / "labels"

OUTPUT_DIR = Path("mmdet_dataset")

CLASS_NAMES = ["lettuce"]
VAL_RATIO = 0.2


def get_all_images():
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(list(IMAGE_DIR.rglob(ext)))
    return sorted(image_files)


def convert_one_split(image_files, split_name):
    out_img_dir = OUTPUT_DIR / "images" / split_name
    out_ann_dir = OUTPUT_DIR / "annotations"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    for i, name in enumerate(CLASS_NAMES):
        coco["categories"].append({
            "id": i + 1,
            "name": name,
            "supercategory": "object"
        })

    image_id = 1
    ann_id = 1

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print("Không đọc được ảnh:", img_path)
            continue

        h, w = img.shape[:2]

        relative_path = img_path.relative_to(IMAGE_DIR)

        safe_name = str(relative_path).replace("\\", "_").replace("/", "_")

        shutil.copy(str(img_path), str(out_img_dir / safe_name))

        coco["images"].append({
            "id": image_id,
            "file_name": safe_name,
            "width": w,
            "height": h
        })

        label_path = LABEL_DIR / relative_path.with_suffix(".txt")

        if not label_path.exists():
            print("Thiếu label:", label_path)
            image_id += 1
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            if len(parts) < 7:
                continue

            class_id = int(float(parts[0]))
            coords = list(map(float, parts[1:]))

            if len(coords) % 2 != 0:
                continue

            polygon = []
            xs = []
            ys = []

            for i in range(0, len(coords), 2):
                x = coords[i] * w
                y = coords[i + 1] * h

                polygon.extend([x, y])
                xs.append(x)
                ys.append(y)

            x_min = min(xs)
            y_min = min(ys)
            x_max = max(xs)
            y_max = max(ys)

            bbox_w = x_max - x_min
            bbox_h = y_max - y_min

            if bbox_w <= 1 or bbox_h <= 1:
                continue

            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": class_id + 1,
                "segmentation": [polygon],
                "bbox": [x_min, y_min, bbox_w, bbox_h],
                "area": bbox_w * bbox_h,
                "iscrowd": 0
            })

            ann_id += 1

        image_id += 1

    json_path = out_ann_dir / f"{split_name}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print(f"Đã tạo: {json_path}")
    print(f"{split_name}: {len(coco['images'])} ảnh, {len(coco['annotations'])} object")


def main():
    image_files = get_all_images()

    if len(image_files) == 0:
        print("Không tìm thấy ảnh trong:", IMAGE_DIR)
        return

    random.seed(42)
    random.shuffle(image_files)

    val_count = int(len(image_files) * VAL_RATIO)

    val_files = image_files[:val_count]
    train_files = image_files[val_count:]

    convert_one_split(train_files, "train")
    convert_one_split(val_files, "val")

    print("XONG!")
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()