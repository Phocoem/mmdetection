"""
Phat hien day du GAP theo dung 4 nhom da quan sat tu heatmap panel 3
(so seed dong gop), roi tu dong sinh phan bu cho nhung gi CO THE sinh
chac chan - phan con lai (danh gia thieu dieu kien trong 1 seed DA
TRAIN ROI) can ban cho biet ten script danh gia tuy chinh cua ban, vi
toi khong biet cach goi dung (khong phai tools/test.py chuan cua
MMDetection, vi no xuat CSV/JSON theo TUNG dieu kien rieng - dac thu
cua pipeline ban tu viet).

=== PHAN LOAI 2 LOAI GAP ===
    A) SEED HOAN TOAN THIEU (khong co thu muc seed_<seed>/ nao) ->
       CAN TRAIN LAI TU DAU. Script nay TU SINH duoc config + lenh chay
       (dung logic generate_seed_configs.py).
    B) SEED DA TON TAI (co thu muc seed_<seed>/) nhung THIEU MOT SO
       DIEU KIEN trong evaluation/ -> chi can chay lai DANH GIA (khong
       can train lai, checkpoint da co san). Script nay CHI liet ke ro
       thieu gi, KHONG tu doan lenh chay vi phu thuoc script danh gia
       rieng cua ban.

Cach dung:
    python fill_seed_gaps.py \\
        --root /home/pc/mmdet_AI/mmdetection/work_dirs/research \\
        --target-seeds 2024 2025 2026 2027 2028 \\
        --configs-dir configs/fair_lettuce \\
        --out-dir gap_fill/
"""
import argparse
import csv
import json
import re
from pathlib import Path

COND_PREFIX = {
    'brightness': 'B', 'contrast': 'C', 'gaussian_noise': 'G',
    'uneven_contrast': 'U', 'dappled_light': 'D',
}
COND_RE = re.compile(
    r'^(brightness|contrast|gaussian_noise|uneven_contrast|dappled_light)_s(\d)$')
SEED_DIR_RE = re.compile(r'^seed_(\d+)$')

CONDITIONS_ORDERED = (
    ['clean'] +
    [f'{fam}_s{s}' for fam in
     ('brightness', 'contrast', 'gaussian_noise',
      'uneven_contrast', 'dappled_light')
     for s in (1, 2, 3)]
)
UNIFORM_CONDS = [c for c in CONDITIONS_ORDERED
                 if c.split('_s')[0] in
                 ('brightness', 'contrast', 'gaussian_noise') and c != 'clean']
SPATIAL_CONDS = [c for c in CONDITIONS_ORDERED
                 if c.split('_s')[0] in ('uneven_contrast', 'dappled_light')]

SEED_CONFIG_TEMPLATE = """# TU DONG SINH boi fill_seed_gaps.py - bu seed hoan toan thieu.
_base_ = '{base_config}'

randomness = dict(seed={seed}, deterministic=True)
work_dir = '{work_dir}/seed_{seed}'
"""


def scan_seed_presence(root: Path, systems=None):
    """Tra ve {system: {seed: set(conditions co san)}} - CHI can biet
    dieu kien nao co, khong can gia tri (khac plot_metrics_heatmap.py)."""
    data = {}
    system_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if systems:
        wanted = set(systems)
        system_dirs = [p for p in system_dirs if p.name in wanted]

    for sys_dir in system_dirs:
        seed_dirs = {SEED_DIR_RE.match(p.name).group(1): p
                     for p in sys_dir.iterdir()
                     if p.is_dir() and SEED_DIR_RE.match(p.name)}
        if not seed_dirs:
            continue
        data[sys_dir.name] = {}
        for seed, seed_dir in seed_dirs.items():
            eval_dir = seed_dir / 'evaluation'
            csv_path = eval_dir / 'condition_metrics.csv'
            conditions_dir = eval_dir / 'conditions'
            found = set()
            if csv_path.exists():
                with open(csv_path, newline='', encoding='utf-8') as f:
                    for row in csv.DictReader(f):
                        if row.get('condition') in CONDITIONS_ORDERED:
                            found.add(row['condition'])
            elif conditions_dir.is_dir():
                for cond_dir in conditions_dir.iterdir():
                    if cond_dir.is_dir() and cond_dir.name in CONDITIONS_ORDERED \
                            and (cond_dir / 'metrics.json').exists():
                        found.add(cond_dir.name)
            data[sys_dir.name][seed] = found
    return data


def analyze_gaps(data: dict, target_seeds):
    """Tra ve list dict mo ta tung gap, phan loai type A/B."""
    gaps = []
    for system, seeds_present in sorted(data.items()):
        for seed in target_seeds:
            seed = str(seed)
            if seed not in seeds_present:
                gaps.append(dict(system=system, seed=seed, type='A_MISSING_SEED',
                                  missing=CONDITIONS_ORDERED[:]))
            else:
                found = seeds_present[seed]
                missing = [c for c in CONDITIONS_ORDERED if c not in found]
                if missing:
                    gaps.append(dict(system=system, seed=seed,
                                      type='B_PARTIAL_EVAL', missing=missing))
    return gaps


def print_group_summary(data: dict, target_seeds):
    """In lai dung 4 nhom da quan sat tu heatmap, tinh tu du lieu that."""
    print('=== PHAN NHOM THEO DO PHU DIEU KIEN (tinh tu du lieu that) ===\n')
    for system, seeds_present in sorted(data.items()):
        uniform_seed_counts = [len(target_seeds and [1]) for _ in ()]
        n_uniform = sum(1 for seed in target_seeds
                         if all(c in seeds_present.get(str(seed), set())
                                for c in UNIFORM_CONDS))
        n_spatial = sum(1 for seed in target_seeds
                         if all(c in seeds_present.get(str(seed), set())
                                for c in SPATIAL_CONDS))
        print(f'{system:<32} uniform du 9 dieu kien: {n_uniform}/{len(target_seeds)} seed'
              f'   |   spatial du 6 dieu kien: {n_spatial}/{len(target_seeds)} seed')
    print()


def write_seed_configs(gaps, base_config_map, configs_dir, out_dir):
    """Voi moi gap type A (seed hoan toan thieu), sinh config override seed
    + gop vao 1 script bash chay tat ca."""
    out_dir.mkdir(parents=True, exist_ok=True)
    launch_lines = ['#!/bin/bash', '# TU DONG SINH boi fill_seed_gaps.py',
                     'FAILED_LOG=launch_missing_seeds_failures.log',
                     '> "$FAILED_LOG"']
    n_generated = 0

    for gap in gaps:
        if gap['type'] != 'A_MISSING_SEED':
            continue
        system, seed = gap['system'], gap['seed']
        if system not in base_config_map:
            print(f'  CANH BAO: khong biet file config goc cua "{system}" '
                  f'(khong co trong --configs-dir hoac khong khop ten) - '
                  f'BO QUA, ban tu tao config nay.')
            continue
        base_config = base_config_map[system]
        content = SEED_CONFIG_TEMPLATE.format(
            base_config=base_config, seed=seed,
            work_dir=f'work_dirs/research/{system}')
        out_path = out_dir / f'{system}_seed{seed}.py'
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)
        launch_lines.append(
            f'if ! python tools/train.py {out_path}; then')
        launch_lines.append(
            f'  echo "THAT BAI (train): {system} seed={seed}" | tee -a "$FAILED_LOG"')
        launch_lines.append('fi')
        n_generated += 1

    launch_lines += [
        '', 'echo ""', 'echo "========================================"',
        'if [ -s "$FAILED_LOG" ]; then',
        '  echo "CO LAN TRAIN THAT BAI - xem chi tiet:"', '  cat "$FAILED_LOG"',
        'else', '  echo "TAT CA LAN TRAIN THANH CONG."', 'fi',
    ]

    launch_path = out_dir / 'launch_missing_seeds.sh'
    with open(launch_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(launch_lines) + '\n')
    print(f'Da sinh {n_generated} config seed con thieu vao {out_dir}')
    print(f'Da sinh script chay: {launch_path}')


def write_partial_report(gaps, out_dir):
    partial = [g for g in gaps if g['type'] == 'B_PARTIAL_EVAL']
    if not partial:
        return
    out_path = out_dir / 'partial_eval_needed.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['System', 'Seed', 'MissingConditions', 'NMissing'])
        for g in partial:
            w.writerow([g['system'], g['seed'], ';'.join(g['missing']),
                        len(g['missing'])])
    print(f'\nDa ghi {len(partial)} dong (seed DA TRAIN nhung thieu danh gia '
          f'mot phan) vao {out_path}')
    print('*** CAC DONG NAY KHONG CAN TRAIN LAI - chi can chay lai DANH GIA ***')
    print('*** Toi CHUA sinh lenh chay cho phan nay vi khong biet ten/cach ***')
    print('*** goi script danh gia tuy chinh cua ban (cai xuat ra ***')
    print('*** condition_metrics.csv / conditions/<cond>/metrics.json). ***')
    print('*** Cho toi biet ten file script do (vd tools/research/eval_conditions.py) ***')
    print('*** va cach goi no, toi sinh chinh xac lenh chay cho tung dong tren. ***')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--target-seeds', nargs='+', type=int,
                         default=[2024, 2025, 2026, 2027, 2028])
    parser.add_argument('--systems', nargs='*', default=None)
    parser.add_argument('--configs-dir', default=None,
                         help='Thu muc chua config goc (vd '
                              'configs/fair_lettuce) - dung de doan file '
                              '_base_ cho tung he thong theo ten khop nhau. '
                              'Bo qua neu chi can xem bao cao, khong can '
                              'sinh config.')
    parser.add_argument('--out-dir', default='gap_fill')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f'Khong tim thay thu muc: {root}')
    out_dir = Path(args.out_dir)

    data = scan_seed_presence(root, args.systems)
    if not data and not args.configs_dir:
        raise SystemExit('Khong doc duoc he thong nao.')

    # SUA LOI: bo sung cac he thong CO CONFIG nhung CHUA TUNG TRAIN (khong
    # co thu muc nao duoi work_dirs/research/<system>/) - neu khong, chung
    # se hoan toan "vo hinh" voi report nay, khong duoc liet ke de bu.
    if args.configs_dir:
        configs_dir_probe = Path(args.configs_dir)
        if configs_dir_probe.is_dir():
            all_config_systems = {
                p.stem for p in configs_dir_probe.glob('*.py')
                if not p.stem.startswith('_')
            }
            if args.systems:
                all_config_systems &= set(args.systems)
            never_trained = sorted(all_config_systems - set(data.keys()))
            for system in never_trained:
                data[system] = {}  # khong seed nao ca -> tat ca target-seeds
                                    # se thanh A_MISSING_SEED cho he thong nay
            if never_trained:
                print(f'CANH BAO: {len(never_trained)} he thong CO CONFIG '
                      f'nhung CHUA TUNG TRAIN (0 seed nao trong work_dirs) - '
                      f'da bo sung vao bao cao: {never_trained}\n')

    if not data:
        raise SystemExit('Khong doc duoc he thong nao.')

    print_group_summary(data, args.target_seeds)

    gaps = analyze_gaps(data, args.target_seeds)
    n_a = sum(1 for g in gaps if g['type'] == 'A_MISSING_SEED')
    n_b = sum(1 for g in gaps if g['type'] == 'B_PARTIAL_EVAL')
    print(f'Tong: {n_a} seed HOAN TOAN THIEU (can train lai), '
          f'{n_b} seed DA TRAIN nhung danh gia thieu mot phan\n')

    base_config_map = {}
    if args.configs_dir:
        configs_dir = Path(args.configs_dir)
        for system in data:
            candidate = configs_dir / f'{system}.py'
            if candidate.exists():
                base_config_map[system] = str(candidate.resolve())
            else:
                print(f'  CANH BAO: khong tim thay {candidate} - he thong '
                      f'"{system}" se bi bo qua o buoc sinh config.')

    out_dir.mkdir(parents=True, exist_ok=True)
    if base_config_map:
        write_seed_configs(gaps, base_config_map, Path(args.configs_dir), out_dir)
    else:
        print('(Chua sinh config train lai vi thieu --configs-dir hop le - '
              'chi in bao cao gap.)')

    write_partial_report(gaps, out_dir)


if __name__ == '__main__':
    main()
