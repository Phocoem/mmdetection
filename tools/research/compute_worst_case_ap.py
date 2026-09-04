"""
Tinh Worst-case AP va bo sung vao bang ket qua (Table 5/6 style).
Khong can GPU - chi xu ly so lieu da co san (CSV xuat tu evaluator).

Cach dung:
    python compute_worst_case_ap.py --input table6_raw.csv \
        --output table6_with_worstcase.csv

Input CSV can cac cot (giong Table 6 trong bai):
    System,Clean,BS1,BS2,BS3,CS1,CS2,CS3,GS1,GS2,GS3,US1,US2,US3,DS1,DS2,DS3,APcorr
(BS=brightness, CS=contrast, GS=gaussian noise, US=uneven contrast,
DS=dappled light)

Output them 3 cot moi:
    WorstAP9   -> min cua 9 dieu kien dong nhat (BS1-3,CS1-3,GS1-3), dung
                  cot nay lam headline vi no cung pham vi voi APcorr da
                  co (Eq. 9 trong bai chi tinh tren 9 dieu kien nay)
    WorstCond9 -> ten dieu kien gay ra worst-case (vd "GS3")
    RD_worst   -> Clean - WorstAP9 (Robustness Drop o kich ban xau nhat)

Dung --include-spatial de tinh tren ca 15 dieu kien (them uneven
contrast/dappled light).
"""
import argparse
import csv

UNIFORM_COLS = ['BS1', 'BS2', 'BS3', 'CS1', 'CS2', 'CS3', 'GS1', 'GS2', 'GS3']
ALL_COND_COLS = UNIFORM_COLS + ['US1', 'US2', 'US3', 'DS1', 'DS2', 'DS3']


def process(input_path: str, output_path: str, include_spatial: bool = False):
    cols = ALL_COND_COLS if include_spatial else UNIFORM_COLS
    suffix = '15' if include_spatial else '9'

    with open(input_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    missing = [c for c in cols if c not in fieldnames]
    if missing:
        raise ValueError(f'Thieu cot trong input CSV: {missing}')

    for row in rows:
        values = {c: float(row[c]) for c in cols}
        worst_cond = min(values, key=values.get)
        worst_ap = values[worst_cond]
        clean = float(row['Clean'])
        row[f'WorstAP{suffix}'] = f'{worst_ap:.3f}'
        row[f'WorstCond{suffix}'] = worst_cond
        row['RD_worst'] = f'{clean - worst_ap:.3f}'

    new_fields = fieldnames + [f'WorstAP{suffix}', f'WorstCond{suffix}',
                                'RD_worst']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Da ghi {len(rows)} dong vao {output_path}\n')
    header = f'{"System":<28}{"APcorr":>8}{"WorstAP" + suffix:>12}' \
             f'{"WorstCond":>12}{"RD_worst":>10}'
    print(header)
    print('-' * len(header))
    for row in rows:
        print(f'{row["System"]:<28}{row.get("APcorr", ""):>8}'
              f'{row[f"WorstAP{suffix}"]:>12}'
              f'{row[f"WorstCond{suffix}"]:>12}'
              f'{row["RD_worst"]:>10}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--include-spatial', action='store_true')
    args = parser.parse_args()
    process(args.input, args.output, args.include_spatial)
