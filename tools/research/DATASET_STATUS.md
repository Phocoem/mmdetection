# Current Dataset Audit

Audit date: 2026-06-15.

## Annotation Integrity

- Train: 477 images, 6,276 annotations.
- Validation: 119 images, 1,568 annotations.
- Test: 712 images, 9,698 annotations.
- One category in every split.
- No duplicate file names.
- No invalid image/category references, invalid boxes, or missing masks.
- No file-name overlap between train, validation, and test.
- Test has more images than train; verify that this split design was intended.

## Blocking Issue

The expected clean directories are missing:

- `mmdet_dataset/lettuce/images/train/`
- `mmdet_dataset/lettuce/images/val/`
- `mmdet_dataset/lettuce/images/test/`

Training and a clean-test corruption benchmark cannot run until these clean
images are restored with the exact file names in the COCO JSON files.

## Existing Legacy Directories

`images/brightness/` and `images/noise/` each contain all 712 test file names,
but they have no source-clean set, severity levels, generation parameters,
random seed, or manifest. They should not be reported as a standardized
corruption benchmark. Do not copy either directory into `images/test/` unless
it is independently verified to contain the original clean test images.

The machine-readable audit is generated at
`work_dirs/research_setup/dataset_audit.json`.

## Earlier Benchmark Toolkit Review

`lettuce_corruption_benchmark/test/` contains 712 PNG files whose stems match
all test annotation file names. Its README describes this directory as clean,
so it is a plausible clean-test candidate, but its original provenance still
needs confirmation before publication.

The earlier toolkit is not used by the new protocol because:

- Its generated `stress/` image folders are currently absent; only metadata
  CSV files remain.
- It uses custom three-level severity settings rather than the established
  five ImageNet-C/COCO-C severity levels.
- Randomness is seeded globally and therefore depends on processing order.
- Every output, including clean, is saved as JPEG, adding recompression.

The new Lettuce-C generator preserves the earlier toolkit untouched and records
stronger seed, dependency, source, annotation, and output hashes.
