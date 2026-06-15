# Protocol nghien cuu instance segmentation rau xa lach

## Nguyen tac chinh

- Chi dung `train` de hoc tham so.
- Chi dung `val` de chon checkpoint, giam learning rate va early stopping.
- Chi danh gia `test` sau khi checkpoint da duoc dong bang.
- Khong chon kien truc, seed, epoch hoac hyperparameter dua tren ket qua test.
- Moi kien truc su dung cung protocol, seed list va corruption manifest.

## Huan luyen den hoi tu

Khong the biet truoc global maximum cua mot neural network. Protocol nay dung:

- Safety cap: 200 epoch.
- Validation moi epoch.
- Primary selection metric: `coco/segm_mAP`.
- Reduce LR 0.5 lan sau 5 epoch plateau.
- Early stop sau 20 epoch khong tang it nhat 0.001 absolute segm mAP.
- Luu `best_coco_segm_mAP_*.pth` va checkpoint cuoi.

Checkpoint tot nhat theo validation, khong phai checkpoint cuoi, duoc dung cho
toan bo clean/corruption test.

## Lettuce-C benchmark

Benchmark duoc sinh mot lan tu clean test bang `imagecorruptions`, thu vien
tham chieu cho ImageNet-C va cac benchmark corruption-C.

Suite noise mac dinh:

- Gaussian noise.
- Shot noise.
- Impulse noise.
- Nam severity level tu 1 den 5.

Suite `label_preserving_c` mo rong gom 13 corruption khong co chu y warp hinh
hoc. `elastic_transform` va `glass_blur` bi loai vi neu noi dung anh bi dich
chuyen ma instance mask khong thay doi, pixel-level ground truth khong con hop
le nghiem ngat.

Moi anh/corruption/severity co seed xac dinh tu SHA-256. Anh output duoc luu
PNG de tranh JPEG recompression khong kiem soat. Manifest luu version thu vien,
seed, annotation hash, source-image hash va output-condition hash.

## Chi so bao cao

Voi primary metric la segmentation AP:

- `P_clean`: AP tren clean test.
- `mPC`: trung binh AP deu tren moi corruption va severity.
- `rPC = mPC / P_clean`: relative performance under corruption.
- `Robustness drop = P_clean - mPC`.

Bao cao them AP theo tung corruption va severity. Moi model can chay it nhat ba
seed da khai bao truoc, sau do bao cao mean va standard deviation. Khi so sanh
hai model, nen dung cung seed va paired statistical test/bootstrap.

Early stopping tao so epoch khac nhau giua cac model. Day la so sanh
convergence-controlled, khong phai compute-matched. Can bao cao stop epoch,
training time, parameter/FLOP va nen them mot ablation compute-matched neu bai
bao dua ra ket luan ve hieu qua tinh toan.

## Log va provenance

Moi run luu:

- Resolved config, seed, git commit/status va source diff.
- Hash annotation train/val/test.
- Dependency versions.
- Console log, JSON scalar va TensorBoard.
- Best/last checkpoint va training summary.
- Metrics cua tung corruption condition.
- Summary clean AP, mPC, rPC va robustness drop.

## Tai lieu nen trich dan

- Hendrycks and Dietterich, *Benchmarking Neural Network Robustness to Common
  Corruptions and Perturbations*, 2019.
- Michaelis et al., *Benchmarking Robustness in Object Detection: Autonomous
  Driving when Winter is Coming*, 2019.
