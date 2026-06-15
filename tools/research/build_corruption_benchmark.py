"""Build a deterministic, label-preserving corruption benchmark."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image


NOISE_CORRUPTIONS = (
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
)

# ImageNet-C/COCO-C corruptions that do not geometrically warp mask labels.
LABEL_PRESERVING_C_CORRUPTIONS = NOISE_CORRUPTIONS + (
    'defocus_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'pixelate',
    'jpeg_compression',
)
MASK_UNSAFE_CORRUPTIONS = {'elastic_transform', 'glass_blur'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source-root',
        default='mmdet_dataset/lettuce',
        help='Root containing annotations/test.json and clean test images.')
    parser.add_argument('--ann-file', default='annotations/test.json')
    parser.add_argument('--image-dir', default='images/test')
    parser.add_argument(
        '--output-root', default='mmdet_dataset/lettuce_c')
    parser.add_argument(
        '--suite',
        choices=('noise', 'label_preserving_c'),
        default='noise')
    parser.add_argument(
        '--corruptions',
        nargs='+',
        help='Explicit imagecorruptions names; overrides --suite.')
    parser.add_argument(
        '--severities', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def condition_seed(global_seed, corruption, severity, file_name):
    text = f'{global_seed}|{corruption}|{severity}|{file_name}'
    return int.from_bytes(
        hashlib.sha256(text.encode('utf-8')).digest()[:4], 'little')


def aggregate_hash(entries):
    digest = hashlib.sha256()
    for name, file_hash in sorted(entries):
        digest.update(name.encode('utf-8'))
        digest.update(file_hash.encode('ascii'))
    return digest.hexdigest()


def main():
    args = parse_args()
    try:
        from imagecorruptions import corrupt
    except ImportError as exc:
        raise ImportError(
            'Install research dependencies first: '
            'pip install -r requirements_research.txt') from exc

    source_root = Path(args.source_root).resolve()
    ann_path = source_root / args.ann_file
    image_dir = source_root / args.image_dir
    output_root = Path(args.output_root).resolve()

    if not ann_path.is_file():
        raise FileNotFoundError(f'Annotation file not found: {ann_path}')
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f'Clean test directory not found: {image_dir}. '
            'Do not generate corruptions from an already corrupted set.')
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f'{output_root} is not empty. Pass --overwrite to rebuild.')
        shutil.rmtree(output_root)

    corruptions = args.corruptions
    suite_name = args.suite if corruptions is None else 'custom'
    if corruptions is None:
        corruptions = (
            NOISE_CORRUPTIONS if args.suite == 'noise'
            else LABEL_PRESERVING_C_CORRUPTIONS)
    unsafe = sorted(set(corruptions) & MASK_UNSAFE_CORRUPTIONS)
    if unsafe:
        raise ValueError(
            f'Corruptions {unsafe} move image content without updating masks '
            'and are disabled for strict instance-segmentation evaluation.')
    severities = sorted(set(args.severities))
    if not severities or any(level not in range(1, 6) for level in severities):
        raise ValueError('Severities must be integers from 1 to 5.')

    annotation = json.loads(ann_path.read_text(encoding='utf-8-sig'))
    images = sorted(annotation['images'], key=lambda item: item['id'])
    output_names = [f'{Path(item["file_name"]).stem}.png' for item in images]
    if len(output_names) != len(set(output_names)):
        raise ValueError('Image file stems are not unique; cannot use PNG names.')

    missing = [
        item['file_name'] for item in images
        if not (image_dir / item['file_name']).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f'{len(missing)} clean test images are missing; first: {missing[0]}')

    output_annotation = json.loads(json.dumps(annotation))
    for item, output_name in zip(output_annotation['images'], output_names):
        item['file_name'] = output_name
    output_ann_path = output_root / 'annotations' / 'test_png.json'
    output_ann_path.parent.mkdir(parents=True, exist_ok=True)
    output_ann_path.write_text(
        json.dumps(output_annotation, ensure_ascii=True),
        encoding='utf-8')

    source_hashes = []
    generated_conditions = []
    for item in images:
        source_path = image_dir / item['file_name']
        source_hashes.append((item['file_name'], sha256_file(source_path)))

    for corruption in corruptions:
        for severity in severities:
            condition_dir = output_root / 'images' / corruption / str(severity)
            condition_dir.mkdir(parents=True, exist_ok=True)
            condition_hashes = []
            for item, output_name in zip(images, output_names):
                source_path = image_dir / item['file_name']
                with Image.open(source_path) as image:
                    rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)

                state = np.random.get_state()
                np.random.seed(
                    condition_seed(
                        args.seed, corruption, severity, item['file_name']))
                try:
                    corrupted = corrupt(
                        rgb,
                        corruption_name=corruption,
                        severity=severity)
                finally:
                    np.random.set_state(state)
                if corrupted.shape != rgb.shape:
                    raise ValueError(
                        f'{corruption} changed image shape from {rgb.shape} '
                        f'to {corrupted.shape}; masks would be invalid.')

                output_path = condition_dir / output_name
                Image.fromarray(
                    np.asarray(corrupted, dtype=np.uint8), mode='RGB').save(
                        output_path, format='PNG', optimize=False)
                condition_hashes.append(
                    (output_name, sha256_file(output_path)))

            generated_conditions.append({
                'corruption': corruption,
                'severity': severity,
                'image_prefix': f'images/{corruption}/{severity}/',
                'image_count': len(images),
                'sha256': aggregate_hash(condition_hashes),
            })
            print(f'Generated {corruption} severity {severity}')

    manifest = {
        'benchmark_name': 'Lettuce-C',
        'suite': suite_name,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'generator': 'imagecorruptions',
        'generator_sha256': sha256_file(Path(__file__).resolve()),
        'dependency_versions': {
            package: version(package)
            for package in (
                'imagecorruptions',
                'numpy',
                'Pillow',
                'scipy',
                'scikit-image',
            )
        },
        'global_seed': args.seed,
        'source_root': str(source_root),
        'source_annotation': args.ann_file,
        'source_annotation_sha256': sha256_file(ann_path),
        'source_image_count': len(images),
        'source_images_sha256': aggregate_hash(source_hashes),
        'output_annotation': 'annotations/test_png.json',
        'output_annotation_sha256': sha256_file(output_ann_path),
        'corruptions': list(corruptions),
        'severities': severities,
        'conditions': generated_conditions,
        'notes': [
            'Only clean test images were used as sources.',
            'Annotations are unchanged except file extensions.',
            'PNG output avoids uncontrolled extra JPEG recompression.',
            'Test corruptions must never be used for model selection.',
        ],
    }
    (output_root / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding='utf-8')
    print(f'Benchmark manifest: {output_root / "manifest.json"}')


if __name__ == '__main__':
    main()
