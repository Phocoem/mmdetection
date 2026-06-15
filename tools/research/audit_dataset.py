"""Audit COCO split integrity and expected clean-image layout."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='mmdet_dataset/lettuce')
    parser.add_argument(
        '--output', default='work_dirs/research_setup/dataset_audit.json')
    parser.add_argument('--check-dimensions', action='store_true')
    parser.add_argument('--hash-images', action='store_true')
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def aggregate_hash(entries):
    digest = hashlib.sha256()
    for name, file_hash in sorted(entries):
        digest.update(name.encode('utf-8'))
        digest.update(file_hash.encode('ascii'))
    return digest.hexdigest()


def audit_split(data_root, split, check_dimensions, hash_images):
    ann_path = data_root / 'annotations' / f'{split}.json'
    image_dir = data_root / 'images' / split
    result = {
        'annotation_path': str(ann_path),
        'image_dir': str(image_dir),
        'errors': [],
        'warnings': [],
    }
    if not ann_path.is_file():
        result['errors'].append('annotation file missing')
        return result, set(), set()

    annotation = json.loads(ann_path.read_text(encoding='utf-8-sig'))
    result['annotation_sha256'] = sha256_file(ann_path)
    images = annotation.get('images', [])
    annotations = annotation.get('annotations', [])
    categories = annotation.get('categories', [])
    image_ids = {item['id'] for item in images}
    category_ids = {item['id'] for item in categories}
    file_names = [item['file_name'] for item in images]
    result.update({
        'num_images': len(images),
        'num_annotations': len(annotations),
        'num_categories': len(categories),
        'unique_file_names': len(set(file_names)),
    })
    if len(file_names) != len(set(file_names)):
        result['errors'].append('duplicate image file names')
    if len(image_ids) != len(images):
        result['errors'].append('duplicate image ids')

    invalid_image_refs = sum(
        item.get('image_id') not in image_ids for item in annotations)
    invalid_category_refs = sum(
        item.get('category_id') not in category_ids for item in annotations)
    invalid_bbox = sum(
        len(item.get('bbox', [])) != 4
        or item.get('bbox', [0, 0, 0, 0])[2] <= 0
        or item.get('bbox', [0, 0, 0, 0])[3] <= 0
        for item in annotations)
    missing_segmentation = sum(
        not item.get('segmentation') for item in annotations)
    result.update({
        'invalid_image_references': invalid_image_refs,
        'invalid_category_references': invalid_category_refs,
        'invalid_bboxes': invalid_bbox,
        'missing_segmentations': missing_segmentation,
    })
    if invalid_image_refs or invalid_category_refs or invalid_bbox:
        result['errors'].append('invalid annotation references or boxes')
    if missing_segmentation:
        result['errors'].append('annotations without segmentation masks')

    if not image_dir.is_dir():
        result['errors'].append('expected clean image directory missing')
        result['missing_images'] = len(images)
        return result, set(file_names), set()

    missing = [
        file_name for file_name in file_names
        if not (image_dir / file_name).is_file()
    ]
    result['missing_images'] = len(missing)
    if missing:
        result['errors'].append(f'{len(missing)} annotated images missing')
        result['first_missing_image'] = missing[0]

    dimension_mismatches = 0
    image_hashes = []
    if check_dimensions or hash_images:
        by_name = {item['file_name']: item for item in images}
        for file_name in file_names:
            path = image_dir / file_name
            if not path.is_file():
                continue
            if check_dimensions:
                with Image.open(path) as image:
                    expected = by_name[file_name]
                    if image.size != (expected['width'], expected['height']):
                        dimension_mismatches += 1
            if hash_images:
                image_hashes.append((file_name, sha256_file(path)))
    if check_dimensions:
        result['dimension_mismatches'] = dimension_mismatches
        if dimension_mismatches:
            result['errors'].append('image dimensions disagree with COCO JSON')
    if hash_images:
        result['image_set_sha256'] = aggregate_hash(image_hashes)
    return result, set(file_names), {item[1] for item in image_hashes}


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    report = {
        'data_root': str(data_root),
        'splits': {},
        'overlaps': {},
        'warnings': [],
    }
    split_files = {}
    split_hashes = {}
    for split in ('train', 'val', 'test'):
        result, files, hashes = audit_split(
            data_root, split, args.check_dimensions, args.hash_images)
        report['splits'][split] = result
        split_files[split] = files
        split_hashes[split] = hashes

    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap = sorted(split_files[left] & split_files[right])
        report['overlaps'][f'{left}_{right}'] = {
            'count': len(overlap),
            'first_file': overlap[0] if overlap else None,
        }
        if overlap:
            report['splits'][left]['errors'].append(
                f'file-name overlap with {right}')
        if args.hash_images:
            content_overlap = split_hashes[left] & split_hashes[right]
            report['overlaps'][f'{left}_{right}'][
                'identical_content_count'] = len(content_overlap)
            if content_overlap:
                report['splits'][left]['errors'].append(
                    f'identical image content overlaps with {right}')

    train_count = report['splits']['train'].get('num_images', 0)
    test_count = report['splits']['test'].get('num_images', 0)
    if test_count > train_count:
        report['warnings'].append(
            'Test split is larger than train split; verify this was intended.')

    legacy_dirs = {}
    test_files = split_files['test']
    for path in sorted((data_root / 'images').iterdir()):
        if not path.is_dir() or path.name in ('train', 'val', 'test'):
            continue
        names = {item.name for item in path.iterdir() if item.is_file()}
        legacy_dirs[path.name] = {
            'image_count': len(names),
            'covers_test_file_names': test_files <= names,
            'has_manifest': (path / 'manifest.json').is_file(),
        }
    report['untracked_image_directories'] = legacy_dirs

    errors = [
        error for result in report['splits'].values()
        for error in result['errors']
    ]
    report['status'] = 'FAIL' if errors else 'PASS'
    report['error_count'] = len(errors)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding='utf-8')
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
