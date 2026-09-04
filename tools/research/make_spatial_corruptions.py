# -*- coding: utf-8 -*-
"""Sinh điều kiện corruption test SPATIALLY-VARYING (không đồng đều theo vùng).

Động cơ: benchmark hiện tại chỉ có contrast/brightness ĐỒNG ĐỀU toàn ảnh.
Nhưng suy giảm thực địa nông nghiệp mang tính KHÔNG ĐỀU theo không gian
(nắng loang lổ qua tán lá, một góc ruộng sáng góc khác trong bóng). BCA
được thiết kế cho biên/cấu trúc, nên sẽ tỏa sáng đúng ở điều kiện không đều
này — nơi augmentation đồng đều generic không có lợi thế đặc thù.

Sinh 2 họ điều kiện mới, mỗi họ 3 severity, LABEL-PRESERVING (chỉ đổi màu,
không đổi hình học nên annotation gốc vẫn đúng):
  - uneven_contrast_s1..s3 : giảm tương phản không đều theo mặt nạ mượt
  - dappled_light_s1..s3   : ánh sáng loang lổ (Perlin-like) qua tán lá

Cách dùng:
    python tools/research/make_spatial_corruptions.py \
        --clean-dir mmdet_dataset/lettuce/test/images \
        --out-root  mmdet_dataset/lettuce_c \
        --seed 0

Kết quả: <out-root>/uneven_contrast_s{1,2,3}/  và  dappled_light_s{1,2,3}/
Sau đó eval như các điều kiện khác bằng evaluate_benchmark.py (annotation
dùng chung với test set gốc).
"""

import argparse
import datetime
import glob
import hashlib
import json
import os

import cv2
import numpy as np


def sha256_of_file(path: str) -> str:
    """Tính sha256 của 1 file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_dir(paths) -> str:
    """Tính sha256 tổng hợp của một tập file (theo thứ tự tên đã sort).

    Băm theo (tên file + nội dung) để dấu vân tay phụ thuộc cả danh sách lẫn
    nội dung -> khớp chuẩn reproducibility như bộ Lettuce-C.
    """
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.basename(p).encode('utf-8'))
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
    return h.hexdigest()


def smooth_random_field(h, w, scale, rng):
    """Sinh trường ngẫu nhiên mượt [0,1] (Perlin-like) bằng upsample noise."""
    small_h = max(2, h // scale)
    small_w = max(2, w // scale)
    noise = rng.random((small_h, small_w)).astype(np.float32)
    field = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(h, w) * 0.01)
    field = (field - field.min()) / (field.max() - field.min() + 1e-6)
    return field


def apply_uneven_contrast(img, severity, rng):
    """Giảm tương phản KHÔNG ĐỀU: vùng theo mặt nạ bị kéo về mức xám cục bộ."""
    h, w = img.shape[:2]
    # severity điều khiển mức giảm tối đa và diện tích ảnh hưởng
    max_reduce = {1: 0.45, 2: 0.65, 3: 0.85}[severity]
    field = smooth_random_field(h, w, scale=8, rng=rng)  # [0,1]
    reduce = (field ** 1.2) * max_reduce               # vùng đậm giảm mạnh
    keep = (1.0 - reduce)[..., None]
    local = cv2.blur(img.astype(np.float32), (41, 41))
    out = keep * img.astype(np.float32) + (1 - keep) * local
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_dappled_light(img, severity, rng):
    """Ánh sáng loang lổ: nhân ảnh với mặt nạ sáng-tối mượt ngẫu nhiên."""
    h, w = img.shape[:2]
    amp = {1: 0.25, 2: 0.40, 3: 0.60}[severity]  # biên độ dao động sáng
    field = smooth_random_field(h, w, scale=6, rng=rng)  # [0,1]
    # gain quanh 1.0: vùng sáng >1, vùng tối <1
    gain = (1.0 - amp) + 2 * amp * field
    out = img.astype(np.float32) * gain[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


CORRUPTIONS = {
    'uneven_contrast': apply_uneven_contrast,
    'dappled_light': apply_dappled_light,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean-dir', required=True,
                        help='Thư mục ảnh test sạch')
    parser.add_argument('--out-root', required=True,
                        help='Thư mục gốc chứa các điều kiện corruption')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    imgs = sorted(
        glob.glob(os.path.join(args.clean_dir, '*.jpg')) +
        glob.glob(os.path.join(args.clean_dir, '*.png')))
    assert imgs, f'Không có ảnh trong {args.clean_dir}'
    print(f'{len(imgs)} ảnh test sạch')

    # Vân tay nguồn (để chứng minh dùng đúng tập ảnh gốc)
    source_images_sha256 = sha256_of_dir(imgs)

    conditions_meta = []
    for name, fn in CORRUPTIONS.items():
        for sev in (1, 2, 3):
            out_dir = os.path.join(args.out_root, f'{name}_s{sev}')
            os.makedirs(out_dir, exist_ok=True)
            written = []
            # seed cố định theo (điều kiện, severity) -> tái lập được, và mỗi
            # ảnh dùng seed riêng ổn định để mọi model test trên CÙNG corruption
            for idx, p in enumerate(imgs):
                rng = np.random.default_rng(
                    args.seed * 100000 + sev * 10000 + idx)
                img = cv2.imread(p)
                out = fn(img, sev, rng)
                dst = os.path.join(out_dir, os.path.basename(p))
                cv2.imwrite(dst, out)
                written.append(dst)
            cond_sha = sha256_of_dir(written)
            conditions_meta.append({
                'corruption': name,
                'severity': sev,
                'image_prefix': f'{name}_s{sev}/',
                'image_count': len(written),
                'sha256': cond_sha,
            })
            print(f'  {name}_s{sev}: {len(written)} ảnh -> {out_dir} '
                  f'(sha256={cond_sha[:12]}...)')

    # Ghi file metadata reproducibility (khớp phong cách bộ Lettuce-C)
    meta = {
        'benchmark_name': 'Lettuce-Spatial',
        'suite': 'label_preserving_spatial',
        'created_at_utc': datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        'generator': 'make_spatial_corruptions.py',
        'generator_sha256': sha256_of_file(os.path.abspath(__file__)),
        'dependency_versions': {
            'numpy': np.__version__,
            'opencv': cv2.__version__,
        },
        'global_seed': args.seed,
        'source_root': os.path.abspath(args.clean_dir),
        'source_image_count': len(imgs),
        'source_images_sha256': source_images_sha256,
        'corruptions': list(CORRUPTIONS.keys()),
        'severities': [1, 2, 3],
        'conditions': conditions_meta,
        'notes': [
            'Spatially-varying corruptions (uneven_contrast, dappled_light).',
            'Annotations unchanged (geometry preserved -> label-preserving).',
            'Per-image deterministic seed -> all models test on identical '
            'corrupted images.',
            'Test corruptions must never be used for model selection.',
        ],
    }
    meta_path = os.path.join(args.out_root, 'benchmark_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\nĐã ghi metadata + sha256: {meta_path}')
    print(f'Xong. Eval bằng evaluate_benchmark.py với --benchmark-root '
          f'{args.out_root} (annotation dùng chung test set gốc).')


if __name__ == '__main__':
    main()
