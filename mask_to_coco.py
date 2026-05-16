import json
import cv2
import numpy as np
import shutil
from pathlib import Path

DATASET_DIR = Path("test")
IMAGE_DIR = DATASET_DIR / "images"
MASK_DIR = DATASET_DIR / "instances"
OUTPUT_DIR = Path("mmdet_dataset")

CLASS_NAMES = ["lettuce"]
MIN_AREA = 10


def get_all_images():
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        files.extend(IMAGE_DIR.rglob(ext))
    return sorted(files)


def find_mask_path(img_path):
    rel = img_path.relative_to(IMAGE_DIR)
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        p = MASK_DIR / rel.with_suffix(ext)
        if p.exists():
            return p
    return None


def read_binary_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if mask is None:
        return None

    # Nếu mask có alpha: dùng kênh alpha
    if len(mask.shape) == 3 and mask.shape[2] == 4:
        gray = mask[:, :, 3]
    # Nếu mask RGB/BGR: đổi sang xám
    elif len(mask.shape) == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask

    # Quan trọng: mọi pixel khác 0 đều là object
    binary = np.where(gray > 0, 255, 0).astype(np.uint8)

    return binary


def mask_to_annotations(mask, image_id, ann_start_id):
    annotations = []
    ann_id = ann_start_id

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print("Số contour tìm được:", len(contours))

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < MIN_AREA:
            continue

        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 3:
            continue

        polygon = approx.reshape(-1, 2).astype(float).flatten().tolist()

        x, y, w, h = cv2.boundingRect(approx)

        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": [polygon],
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": float(area),
            "iscrowd": 0
        })

        ann_id += 1

    return annotations, ann_id


def convert_dataset():
    out_img_dir = OUTPUT_DIR / "images"
    out_ann_dir = OUTPUT_DIR / "annotations"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": CLASS_NAMES[0], "supercategory": "object"}
        ]
    }

    image_id = 1
    ann_id = 1

    for img_path in get_all_images():
        print("\nẢnh:", img_path)

        img = cv2.imread(str(img_path))
        if img is None:
            print("Không đọc được ảnh")
            continue

        h, w = img.shape[:2]

        mask_path = find_mask_path(img_path)
        if mask_path is None:
            print("Không tìm thấy mask tương ứng")
            continue

        print("Mask:", mask_path)

        mask = read_binary_mask(mask_path)
        if mask is None:
            print("Không đọc được mask")
            continue

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        print("Pixel object:", np.count_nonzero(mask))

        relative_path = img_path.relative_to(IMAGE_DIR)

        safe_name = str(relative_path) \
            .replace("\\", "_") \
            .replace("/", "_")

        shutil.copy(str(img_path), str(out_img_dir / safe_name))


        coco["images"].append({
            "id": image_id,
            "file_name": safe_name,
            "width": w,
            "height": h
        })

        anns, ann_id = mask_to_annotations(mask, image_id, ann_id)
        coco["annotations"].extend(anns)

        image_id += 1

    json_path = out_ann_dir / "test.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print("\nĐã tạo:", json_path)
    print("Số ảnh:", len(coco["images"]))
    print("Số object:", len(coco["annotations"]))


if __name__ == "__main__":
    convert_dataset()