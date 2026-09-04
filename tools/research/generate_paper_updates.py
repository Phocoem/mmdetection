"""
Tinh TOAN BO con so can thay trong ban thao moi nhat (ban da sua loi trinh
bay, con dung du lieu 3-seed) - doc truc tiep tu per_seed_full.csv, sinh
ra 1 file bao cao anh xa DUNG tung cau trong bai voi gia tri 5-seed moi.

Cach dung:
    python generate_paper_updates.py --input per_seed_full.csv \\
        --out paper_updates.txt
"""
import argparse
import csv
import statistics
from pathlib import Path

from scipy import stats

UNIFORM = ['BS1', 'BS2', 'BS3', 'CS1', 'CS2', 'CS3', 'GS1', 'GS2', 'GS3']
ALL_COND = UNIFORM + ['US1', 'US2', 'US3', 'DS1', 'DS2', 'DS3']
FAMILIES = {
    'Bright.': ['BS1', 'BS2', 'BS3'], 'Contrast': ['CS1', 'CS2', 'CS3'],
    'Gauss.': ['GS1', 'GS2', 'GS3'], 'Uneven': ['US1', 'US2', 'US3'],
    'Dappled': ['DS1', 'DS2', 'DS3'],
}


def load(path):
    data = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            system, seed = row['System'], row['Seed']
            vals = {c: float(row[c]) for c in ['Clean'] + ALL_COND
                    if c in row and row[c] not in (None, '')}
            data.setdefault(system, {})[seed] = vals
    return data


def apcorr_per_seed(seed_vals):
    if not all(c in seed_vals for c in UNIFORM):
        return None
    return statistics.mean(seed_vals[c] for c in UNIFORM)


def family_per_seed(seed_vals, cols):
    if not all(c in seed_vals for c in cols):
        return None
    return statistics.mean(seed_vals[c] for c in cols)


def paired_ttest(a, b):
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    mean_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs) if n > 1 else 0.0
    if std_diff == 0:
        return dict(n=n, mean_diff=mean_diff, std_diff=0.0,
                     t=float('inf') if mean_diff else 0.0, df=n - 1,
                     p=0.0 if mean_diff else 1.0)
    se = std_diff / (n ** 0.5)
    t = mean_diff / se
    df = n - 1
    p = float(stats.t.sf(abs(t), df) * 2)
    return dict(n=n, mean_diff=mean_diff, std_diff=std_diff, t=t, df=df, p=p)


def compare(data, sys_a, sys_b, metric_fn, label):
    common = sorted(set(data.get(sys_a, {})) & set(data.get(sys_b, {})))
    a_vals, b_vals = [], []
    for s in common:
        va, vb = metric_fn(data[sys_a][s]), metric_fn(data[sys_b][s])
        if va is None or vb is None:
            continue
        a_vals.append(va)
        b_vals.append(vb)
    if len(a_vals) < 2:
        return f'{label}: KHONG DU DU LIEU (chi {len(a_vals)} seed chung)'
    r = paired_ttest(a_vals, b_vals)
    return (f'{label}\n'
            f'    {sys_a} mean={statistics.mean(a_vals):.4f}  '
            f'{sys_b} mean={statistics.mean(b_vals):.4f}\n'
            f'    diff(A-B)={r["mean_diff"]:+.4f}  t={r["t"]:.2f}  '
            f'df={r["df"]}  p={r["p"]:.4f}  n={r["n"]}')


def system_mean(data, system, col_or_fn):
    seeds = data.get(system, {})
    vals = [col_or_fn(v) for v in seeds.values()]
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def system_std(data, system, metric_fn):
    seeds = data.get(system, {})
    vals = [metric_fn(v) for v in seeds.values()]
    vals = [v for v in vals if v is not None]
    return statistics.stdev(vals) if len(vals) > 1 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    data = load(args.input)
    out = []

    def emit(s=''):
        out.append(s)

    emit('BAO CAO CAP NHAT SO LIEU - BAN THAO MOI NHAT (5-seed thay 3-seed)')
    emit('=' * 78)

    # ---- Table 5 --------------------------------------------------------
    emit('\n### TABLE 5 (Main comparison) - gia tri moi tung he thong ###')
    t5_systems = [
        ('mask_rcnn_r50_fpn', 'Mask R-CNN R50 (FPN)'),
        ('mask_rcnn_r101_fpn', 'Mask R-CNN R101'),
        ('mask_rcnn_r50_cbam_fpn', 'CBAM-FPN'),
        ('mask_rcnn_r50_bifpn', 'BiFPN'),
        ('mask_rcnn_r50_fpn_aug', 'CPU photometric aug'),
        ('mask_rcnn_r50_gpuaug', 'IAPC control (l=0)'),
        ('mask_rcnn_r50_iapc_lam0p25', 'IAPC (l=0.25)'),
    ]
    for key, label in t5_systems:
        clean = system_mean(data, key, lambda v: v.get('Clean'))
        row = [label, f'Clean={clean:.3f}' if clean else 'Clean=?']
        for fam, cols in FAMILIES.items():
            m = system_mean(data, key, lambda v, c=cols: family_per_seed(v, c))
            row.append(f'{fam}={m:.3f}' if m is not None else f'{fam}=?')
        apc = system_mean(data, key, apcorr_per_seed)
        row.append(f'APcorr={apc:.3f}' if apc is not None else 'APcorr=?')
        emit('  ' + '  '.join(row))

    # ---- Section 4.3 Tier 1 ---------------------------------------------
    emit('\n### SECTION 4.3 (Tier 1 - Presence of Augmentation) ###')
    emit('So goc: "raising mean corrupted mask AP from 0.617 to 0.709", '
         '"robustness drop from 0.113 to 0.021", "t=59.4 tren contrast S3, '
         't=21.1 tren Gaussian noise"')
    fpn_apc = system_mean(data, 'mask_rcnn_r50_fpn', apcorr_per_seed)
    aug_apc = system_mean(data, 'mask_rcnn_r50_fpn_aug', apcorr_per_seed)
    fpn_clean = system_mean(data, 'mask_rcnn_r50_fpn', lambda v: v.get('Clean'))
    aug_clean = system_mean(data, 'mask_rcnn_r50_fpn_aug', lambda v: v.get('Clean'))
    emit(f'  APcorr: {fpn_apc:.4f} -> {aug_apc:.4f}')
    emit(f'  RD: {fpn_clean - fpn_apc:.4f} -> {aug_clean - aug_apc:.4f}')
    emit('  ' + compare(data, 'mask_rcnn_r50_fpn_aug', 'mask_rcnn_r50_fpn',
                         apcorr_per_seed, 'Tier1 tren APcorr tong the'))
    emit('  ' + compare(data, 'mask_rcnn_r50_fpn_aug', 'mask_rcnn_r50_fpn',
                         lambda v: v.get('CS3'), 'Tier1 tren dieu kien CS3 '
                         '(thay cho "t=59.4" trong ban goc)'))
    emit('  ' + compare(data, 'mask_rcnn_r50_fpn_aug', 'mask_rcnn_r50_fpn',
                         lambda v: v.get('GS3'), 'Tier1 tren dieu kien GS3 '
                         '(thay cho "t=21.1" trong ban goc)'))

    # ---- Section 4.4 Tier 2 ----------------------------------------------
    emit('\n### SECTION 4.4 (Tier 2 - Neck/Backbone Variant) ###')
    emit('So goc: "entire span of APcorr is 0.042, from 0.607 to 0.649", '
         '"ResNet-101 ... lower corrupted AP (0.607 versus 0.617)", '
         '"Augmentation outperforms strongest architectural variant by 0.060 AP"')
    for key in ['mask_rcnn_r50_fpn', 'mask_rcnn_r101_fpn',
                'mask_rcnn_r50_cbam_fpn', 'mask_rcnn_r50_bifpn']:
        v = system_mean(data, key, apcorr_per_seed)
        emit(f'  APcorr {key} = {v:.4f}' if v is not None else f'  {key}: khong co du lieu')
    bifpn_apc = system_mean(data, 'mask_rcnn_r50_bifpn', apcorr_per_seed)
    emit(f'  Aug vuot BiFPN: {aug_apc - bifpn_apc:.4f} AP (ban goc: 0.060)')
    emit('  ' + compare(data, 'mask_rcnn_r50_bifpn', 'mask_rcnn_r50_fpn',
                         apcorr_per_seed, 'BiFPN vs FPN (t-test, ban goc CHUA co)'))

    # ---- Section 4.5 Tier 3 -----------------------------------------------
    emit('\n### SECTION 4.5 (Tier 3 - Augmentation Recipe) ###')
    emit('So goc: "three augmentation recipes ... 0.709, 0.699, 0.703 -> '
         'span 0.011", "sampler gains +0.009 AP tren GS3 (t=4.90), loses '
         '-0.088 AP tren CS3 (t=-4.87)"')
    gpuaug_apc = system_mean(data, 'mask_rcnn_r50_gpuaug', apcorr_per_seed)
    cpuaug_apc = system_mean(data, 'mask_rcnn_r50_iapc_cpuaug', apcorr_per_seed)
    emit(f'  APcorr: CPU aug={aug_apc:.4f}, GPU control={gpuaug_apc:.4f}, '
         f'IAPC+CPUaug={cpuaug_apc:.4f}')
    emit('  ' + compare(data, 'mask_rcnn_r50_gpuaug', 'mask_rcnn_r50_fpn_aug',
                         lambda v: v.get('GS3'),
                         'Sampler(GPU) vs CPU tren GS3 (thay "+0.009,t=4.90")'))
    emit('  ' + compare(data, 'mask_rcnn_r50_gpuaug', 'mask_rcnn_r50_fpn_aug',
                         lambda v: v.get('CS3'),
                         'Sampler(GPU) vs CPU tren CS3 (thay "-0.088,t=-4.87")'))
    emit('  *** CAN THEM doan sua "hai khac biet" -> "bon khac biet" '
         '(xem van_ban_can_sua.txt muc 1 - corruption_cfg khong cong bang) ***')

    # ---- Section 4.6 Tier 4 -----------------------------------------------
    emit('\n### SECTION 4.6 (Tier 4 - Feature-Level Constraints) ###')
    emit('So goc: "+0.001 +-0.005 AP tren Gaussian family (t=0.48)", '
         '"+0.029 +-0.036 tai contrast S3 (t=1.37)", "family-level mean '
         'tren contrast la +0.014", "-0.000 +-0.006 tren brightness (t=-0.13)"')
    emit('  ' + compare(data, 'mask_rcnn_r50_iapc_lam0p25', 'mask_rcnn_r50_gpuaug',
                         apcorr_per_seed, 'IAPC(0.25) vs control - APcorr tong the'))
    emit('  ' + compare(data, 'mask_rcnn_r50_iapc_lam0p25', 'mask_rcnn_r50_gpuaug',
                         lambda v: v.get('GS3'), 'tren GS3 (thay "t=0.48")'))
    emit('  ' + compare(data, 'mask_rcnn_r50_iapc_lam0p25', 'mask_rcnn_r50_gpuaug',
                         lambda v: v.get('CS3'), 'tren CS3 (thay "t=1.37")'))
    emit('  ' + compare(data, 'mask_rcnn_r50_iapc_lam0p25', 'mask_rcnn_r50_gpuaug',
                         lambda v: family_per_seed(v, FAMILIES['Contrast']),
                         'tren trung binh family Contrast (thay "+0.014")'))
    emit('  ' + compare(data, 'mask_rcnn_r50_iapc_lam0p25', 'mask_rcnn_r50_gpuaug',
                         lambda v: v.get('BS3'), 'tren BS3 (thay "t=-0.13")'))
    l1_apc = system_mean(data, 'mask_rcnn_r50_iapc', apcorr_per_seed)
    emit(f'  APcorr lambda=1.0 (mask_rcnn_r50_iapc) = {l1_apc:.4f} '
         f'(ban goc: 0.694, control: 0.699) -> van thap hon control: '
         f'{"CO" if l1_apc < gpuaug_apc else "KHONG"}')

    # ---- Section 4.9 Stability --------------------------------------------
    emit('\n### SECTION 4.9 (Stability Across Seeds) ###')
    emit('So goc: "std 0.008 IAPC, 0.009 augmentation, 0.025 CBAM-FPN"')
    for key, name in [('mask_rcnn_r50_iapc_lam0p25', 'IAPC(0.25)'),
                       ('mask_rcnn_r50_fpn_aug', 'FPN+Aug'),
                       ('mask_rcnn_r50_cbam_fpn', 'CBAM-FPN')]:
        std = system_std(data, key, apcorr_per_seed)
        emit(f'  Std APcorr {name} = {std:.4f}' if std is not None else f'  {name}: N/A')

    # ---- Section 4.10 Lambda sweep ------------------------------------------
    emit('\n### SECTION 4.10 (Sensitivity to lambda) ###')
    for key, lam in [('mask_rcnn_r50_gpuaug', '0'),
                      ('mask_rcnn_r50_iapc_lam0p10', '0.1'),
                      ('mask_rcnn_r50_iapc_lam0p25', '0.25'),
                      ('mask_rcnn_r50_iapc_lam0p50', '0.5'),
                      ('mask_rcnn_r50_iapc', '1.0')]:
        apc = system_mean(data, key, apcorr_per_seed)
        emit(f'  lambda={lam}: APcorr={apc:.4f}' if apc is not None else f'  lambda={lam}: N/A')

    # ---- Table 7 (Component Ablation) --------------------------------------
    emit('\n### TABLE 7 (Component Ablation) ###')
    for key, label in [
        ('mask_rcnn_r50_iapc_lam0p25', 'IAPC (full)'),
        ('mask_rcnn_r50_abl_global', 'w/o instance-awareness'),
        ('mask_rcnn_r50_abl_p2only', 'P2 only'),
        ('mask_rcnn_r50_abl_cosonly', 'cosine only (a=1)'),
        ('mask_rcnn_r50_abl_l1only', 'l1 only (a=0)'),
        ('mask_rcnn_r50_abl_nosg', 'w/o stop-gradient'),
    ]:
        clean = system_mean(data, key, lambda v: v.get('Clean'))
        apc = system_mean(data, key, apcorr_per_seed)
        emit(f'  {label}: Clean={clean:.3f}  APcorr={apc:.3f}'
             if clean and apc else f'  {label}: N/A')

    Path(args.out).write_text('\n'.join(out), encoding='utf-8')
    print(f'Da ghi bao cao vao {args.out} ({len(out)} dong)')
    print('\n'.join(out[:20]))
    print('... (xem file day du)')


if __name__ == '__main__':
    main()
