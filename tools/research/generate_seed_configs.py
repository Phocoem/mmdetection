"""
Sinh cau hinh cho nhieu seed doc lap, dam bao KHONG con loi "seed hardcode
dung chung" da phat hien trong fair_protocol.py (fair_randomness=seed=2025
ap dung chung cho moi he thong qua _base_, khong thay file override rieng
cho seed 2024/2026).

Dung cho CA HAI muc dich:
  (a) SUA lai 3 seed goc (2024, 2025, 2026) cho 16 he thong - dam bao moi
      seed thuc su doc lap, khong bi trung lang le.
  (b) MO RONG len 5-8 seed rieng cho Tier 3/Tier 4 (augmentation recipe,
      feature-constraint) de tang power thong ke (checklist muc #10).

Cach dung:
    python generate_seed_configs.py \\
        --base-config configs/research/mask_rcnn_r50_iapc.py \\
        --seeds 2024 2025 2026 2027 2028 \\
        --out-dir configs/research/seed_variants/

Sinh ra N file config (moi file override randomness rieng, KHONG ke thua
gia tri hardcode tu fair_protocol.py nua) + 1 script bash chay lan luot.
"""
import argparse
import os
import re

TEMPLATE = """# TU DONG SINH boi generate_seed_configs.py - KHONG sua tay.
# fair_protocol.py hardcode randomness.seed={base_seed} dung chung cho moi
# he thong qua _base_ - file nay OVERRIDE rieng de dam bao seed={seed}
# thuc su duoc ap dung doc lap, khong bi ke thua nham gia tri {base_seed}.
_base_ = '{base_config}'

randomness = dict(seed={seed}, deterministic=True)
work_dir = '{work_dir}/seed_{seed}'
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-config', required=True,
                         help='Duong dan config goc (IAPC, FPN+Aug, ...)')
    parser.add_argument('--seeds', nargs='+', type=int, required=True,
                         help='Danh sach seed, vd: 2024 2025 2026 2027 2028')
    parser.add_argument('--out-dir', required=True,
                         help='Thu muc luu cac file config da sinh')
    parser.add_argument('--work-dir-root', default=None,
                         help='Goc work_dir (mac dinh suy tu ten base config)')
    parser.add_argument('--base-seed-in-config', type=int, default=2025,
                         help='Gia tri seed hardcode dang co trong '
                              'fair_protocol.py, chi de ghi chu canh bao '
                              'trong file sinh ra')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stem = re.sub(r'\.py$', '', os.path.basename(args.base_config))
    work_dir_root = args.work_dir_root or f'work_dirs/research/{stem}'
    base_config_abs = os.path.abspath(args.base_config)  # SUA: tranh loi _base_
                                                           # bi resolve sai khi
                                                           # file sinh ra nam
                                                           # trong --out-dir khac
                                                           # thu muc goc project

    generated = []
    for seed in args.seeds:
        content = TEMPLATE.format(
            base_config=base_config_abs,
            base_seed=args.base_seed_in_config,
            seed=seed,
            work_dir=work_dir_root)
        out_path = os.path.join(args.out_dir, f'{stem}_seed{seed}.py')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)
        generated.append(out_path)
        print(f'Da sinh: {out_path}')

    launch_script = os.path.join(args.out_dir, f'launch_{stem}.sh')
    with open(launch_script, 'w', encoding='utf-8') as f:
        f.write('#!/bin/bash\n')
        f.write('# TU DONG SINH - chay lan luot tat ca seed vua tao\n')
        f.write('# KHONG dung set -e - neu 1 seed loi, cac seed con lai van\n')
        f.write('# tiep tuc chay, loi duoc ghi vao FAILED_LOG.\n')
        f.write(f'FAILED_LOG={stem}_train_failures.log\n')
        f.write('> "$FAILED_LOG"\n')
        for p in generated:
            f.write(f'if ! python tools/train.py {p}; then\n')
            f.write(f'  echo "THAT BAI (train): {p}" | tee -a "$FAILED_LOG"\n')
            f.write('fi\n')
        f.write('echo ""\n')
        f.write('if [ -s "$FAILED_LOG" ]; then\n')
        f.write('  echo "CO SEED THAT BAI - xem chi tiet:"; cat "$FAILED_LOG"\n')
        f.write('else\n')
        f.write('  echo "TAT CA SEED TRAIN THANH CONG."\n')
        f.write('fi\n')
    os.chmod(launch_script, 0o755)
    print(f'Da sinh script chay: {launch_script}')


if __name__ == '__main__':
    main()
