# -*- coding: utf-8 -*-
"""Paired bootstrap 95% CI + significance test giữa 2 mô hình (MC7).

Nguyên lý: resample ảnh test CÓ HOÀN LẠI (with replacement), tính lại segm mAP
cho CẢ HAI mô hình trên cùng mẫu bootstrap (paired), lặp B lần:
- 95% CI của mAP mỗi mô hình = percentile [2.5, 97.5] của phân phối bootstrap
- 95% CI của hiệu (A - B)
- p-value (2 phía) = 2 * min(P(diff <= 0), P(diff >= 0))

Kỹ thuật: COCOeval loại bỏ imgIds trùng lặp, nên khi resample có hoàn lại,
script tạo bản sao ảnh + annotation + prediction với image_id MỚI cho mỗi lần
ảnh được rút — đây là cách paired bootstrap đúng chuẩn cho COCO AP.

Chuẩn bị input — dump prediction ra JSON bằng mmdet:
    python tools/test.py CFG CKPT \
        --cfg-options test_evaluator.outfile_prefix=preds/model_a_clean
    -> tạo preds/model_a_clean.segm.json

Cách dùng:
    python tools/research/paired_bootstrap_test.py \
        --ann mmdet_dataset/lettuce/annotations/test.json \
        --pred-a preds/dgcf_v2_clean.segm.json \
        --pred-b preds/fpn_clean.segm.json \
        --name-a DGCF-FPNv2 --name-b MaskRCNN-R50 \
        --num-bootstrap 500 --out bootstrap_clean.json

Thời gian tham khảo: ~2-4s/lần eval x 2 model x B=500 ~ 30-60 phút/điều kiện.
"""

import argparse
import contextlib
import copy
import io
import json
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_gt(ann_file):
    with open(ann_file) as f:
        gt = json.load(f)
    return gt


def index_by_image(items, key='image_id'):
    idx = defaultdict(list)
    for it in items:
        idx[it[key]].append(it)
    return idx


def build_bootstrap_sample(gt, preds_a, preds_b, sampled_ids):
    """Tạo gt/pred mới với image_id mới cho từng lần rút (kể cả trùng)."""
    img_index = {im['id']: im for im in gt['images']}
    ann_index = index_by_image(gt['annotations'])
    pa_index = index_by_image(preds_a)
    pb_index = index_by_image(preds_b)

    new_images, new_anns, new_pa, new_pb = [], [], [], []
    next_ann_id = 1
    for new_id, old_id in enumerate(sampled_ids, start=1):
        im = dict(img_index[old_id])
        im['id'] = new_id
        new_images.append(im)
        for a in ann_index.get(old_id, []):
            a2 = dict(a)
            a2['id'] = next_ann_id
            next_ann_id += 1
            a2['image_id'] = new_id
            new_anns.append(a2)
        for p in pa_index.get(old_id, []):
            p2 = dict(p)
            p2['image_id'] = new_id
            new_pa.append(p2)
        for p in pb_index.get(old_id, []):
            p2 = dict(p)
            p2['image_id'] = new_id
            new_pb.append(p2)

    new_gt = {'images': new_images, 'annotations': new_anns,
              'categories': copy.deepcopy(gt['categories'])}
    return new_gt, new_pa, new_pb


def eval_map(gt_dict, preds):
    if not preds:
        return 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = gt_dict
        coco.createIndex()
        dt = coco.loadRes(preds)
        ev = COCOeval(coco, dt, iouType='segm')
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[0])  # mAP @[0.5:0.95]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', required=True)
    parser.add_argument('--pred-a', required=True)
    parser.add_argument('--pred-b', required=True)
    parser.add_argument('--name-a', default='model_A')
    parser.add_argument('--name-b', default='model_B')
    parser.add_argument('--num-bootstrap', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='bootstrap_result.json')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    gt = load_gt(args.ann)
    with open(args.pred_a) as f:
        preds_a = json.load(f)
    with open(args.pred_b) as f:
        preds_b = json.load(f)

    img_ids = [im['id'] for im in gt['images']]
    n = len(img_ids)
    print(f'{n} ảnh test, B={args.num_bootstrap} bootstrap resamples')

    # Điểm trên toàn bộ test set (point estimate)
    full_a = eval_map(copy.deepcopy(
        {'images': gt['images'], 'annotations': gt['annotations'],
         'categories': gt['categories']}), preds_a)
    full_b = eval_map(copy.deepcopy(
        {'images': gt['images'], 'annotations': gt['annotations'],
         'categories': gt['categories']}), preds_b)
    print(f'Point estimate: {args.name_a}={full_a:.4f}, '
          f'{args.name_b}={full_b:.4f}, diff={full_a - full_b:+.4f}')

    maps_a, maps_b = [], []
    for b in range(args.num_bootstrap):
        sampled = rng.choice(img_ids, size=n, replace=True)
        bgt, bpa, bpb = build_bootstrap_sample(gt, preds_a, preds_b, sampled)
        maps_a.append(eval_map(bgt, bpa))
        maps_b.append(eval_map(copy.deepcopy(bgt), bpb))
        if (b + 1) % 50 == 0:
            d = np.array(maps_a) - np.array(maps_b)
            print(f'  [{b + 1}/{args.num_bootstrap}] '
                  f'diff mean={d.mean():+.4f}')

    a = np.array(maps_a)
    bb = np.array(maps_b)
    d = a - bb

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    p_le = float((d <= 0).mean())
    p_ge = float((d >= 0).mean())
    p_two_sided = float(min(1.0, 2 * min(p_le, p_ge)))

    result = {
        'n_images': n,
        'num_bootstrap': args.num_bootstrap,
        args.name_a: {'point': full_a, 'boot_mean': float(a.mean()),
                      'boot_std': float(a.std()), 'ci95': ci(a)},
        args.name_b: {'point': full_b, 'boot_mean': float(bb.mean()),
                      'boot_std': float(bb.std()), 'ci95': ci(bb)},
        'diff_A_minus_B': {'point': full_a - full_b,
                           'boot_mean': float(d.mean()),
                           'boot_std': float(d.std()), 'ci95': ci(d)},
        'p_value_two_sided': p_two_sided,
        'significant_at_0.05': bool(p_two_sided < 0.05),
    }
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)

    print('\n===== KẾT QUẢ =====')
    print(f"{args.name_a}: {full_a:.4f}  CI95={result[args.name_a]['ci95']}")
    print(f"{args.name_b}: {full_b:.4f}  CI95={result[args.name_b]['ci95']}")
    print(f"Diff: {full_a - full_b:+.4f}  "
          f"CI95={result['diff_A_minus_B']['ci95']}")
    print(f"p-value (2 phía): {p_two_sided:.4f}  "
          f"{'CÓ' if p_two_sided < 0.05 else 'KHÔNG'} ý nghĩa ở mức 0.05")
    print(f'Đã lưu: {args.out}')


if __name__ == '__main__':
    main()
