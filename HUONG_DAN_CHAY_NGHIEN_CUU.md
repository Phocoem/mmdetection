# Huong dan chay nghien cuu Lettuce Instance Segmentation

Tai lieu nay mo ta quy trinh chuan de huan luyen, chon checkpoint, tao benchmark
noise va danh gia cac kien truc trong `configs/fair_lettuce/`.

## 1. Mo PowerShell tai thu muc du an

```powershell
cd C:\Users\ADMIN\Desktop\Dead15_5\mmdetection
```

Su dung Python trong moi truong da cai MMDetection:

```powershell
$PYTHON = "..\mmdet_env\Scripts\python.exe"
```

## 2. Kiem tra cau truc du lieu

Du lieu clean bat buoc phai co cau truc:

```text
mmdet_dataset/lettuce/
  annotations/train.json
  annotations/val.json
  annotations/test.json
  images/train/
  images/val/
  images/test/
```

Chay audit truoc khi huan luyen:

```powershell
& $PYTHON tools\research\audit_dataset.py --check-dimensions --hash-images
```

Khong tiep tuc huan luyen neu audit bao thieu anh, sai kich thuoc, trung anh
giua cac split hoac annotation khong hop le.

## 3. Tao benchmark noise tu clean test

Benchmark noise khuyen nghi gom Gaussian noise, shot noise va impulse noise,
moi loai co 5 muc severity:

```powershell
& $PYTHON tools\research\build_corruption_benchmark.py --suite noise
```

Tao benchmark corruption mo rong:

```powershell
& $PYTHON tools\research\build_corruption_benchmark.py --suite label_preserving_c --overwrite
```

Moi benchmark deu luu seed, tham so, hash anh nguon va hash anh dau ra trong
manifest. Khong thay doi mask cua tap test.

## 4. Kiem tra tinh fair cua config

```powershell
& $PYTHON configs\fair_lettuce\audit_fairness.py
```

Ket qua mong doi:

```text
PASS: 16 configs share the same protocol (static audit).
```

## 5. Chay mot mo hinh

Huan luyen va danh gia mot kien truc voi seed co dinh:

```powershell
& $PYTHON tools\research\run_experiment.py `
  configs\fair_lettuce\mask_rcnn_r50_fpn.py `
  --seed 2026 `
  --amp
```

Qua trinh huan luyen:

- Chon checkpoint bang validation segmentation mAP.
- Giam learning rate khi validation mAP khong cai thien.
- Dung som sau 20 validation epoch khong tang it nhat `0.001` mAP.
- Co gioi han an toan toi da 200 epoch.
- Danh gia cung mot best checkpoint tren clean test va tat ca noise.

## 6. Chay toan bo nghien cuu

Lenh sau chay tat ca 16 kien truc voi ba seed da khai bao truoc:
`2026`, `2027`, `2028`.

```powershell
& $PYTHON tools\research\run_study.py --amp
```

Qua trinh nay co the mat nhieu thoi gian. Khong thay doi seed hoac config sau
khi da xem ket qua test.

## 7. Tong hop ket qua

Sau khi cac run hoan thanh:

```powershell
& $PYTHON tools\research\summarize_study.py
```

Bao cao tong hop gom:

- Clean segmentation mAP.
- Mean Performance under Corruption (`mPC`).
- Relative Performance under Corruption (`rPC`).
- Robustness drop.
- Mean va standard deviation qua cac seed.
- Epoch dung, thoi gian huan luyen va thoi gian danh gia.

## 8. Thu muc ket qua

Ket qua duoc luu trong `work_dirs/research/`. Moi run bao gom:

```text
console log
resolved config
JSON va TensorBoard curves
best va last checkpoint
dependency versions
Git commit va source diff
metrics cua tung corruption
summary JSON va CSV
```

Mo TensorBoard:

```powershell
& $PYTHON -m tensorboard.main --logdir work_dirs\research
```

## 9. Nguyen tac bao cao trong bai nghien cuu

1. Dung validation set de chon checkpoint va dung som.
2. Chi dung test set de bao cao ket qua cuoi.
3. Dung cung mot corruption manifest cho moi kien truc.
4. Bao cao mean va standard deviation cua it nhat ba seed.
5. Bao cao ca clean AP, mPC, rPC va robustness drop.
6. Bao cao epoch hoi tu, thoi gian va chi phi tinh toan.
7. Khong coi early stopping la bang chung dat global maximum.

## 10. Lenh chay nhanh day du

```powershell
cd C:\Users\ADMIN\Desktop\Dead15_5\mmdetection
$PYTHON = "..\mmdet_env\Scripts\python.exe"

& $PYTHON tools\research\audit_dataset.py --check-dimensions --hash-images
& $PYTHON tools\research\build_corruption_benchmark.py --suite noise
& $PYTHON configs\fair_lettuce\audit_fairness.py
& $PYTHON tools\research\run_study.py --amp
& $PYTHON tools\research\summarize_study.py
```

Neu buoc audit dataset that bai, phai sua du lieu clean truoc khi chay cac lenh
con lai.
