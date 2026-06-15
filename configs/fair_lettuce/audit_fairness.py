"""Verify every experiment points to the same controlled protocol."""

import argparse
import ast
from copy import deepcopy
from pathlib import Path


PROTOCOL_ASSIGNMENTS = {
    'train_dataloader': 'fair_train_dataloader',
    'val_dataloader': 'fair_val_dataloader',
    'test_dataloader': 'fair_test_dataloader',
    'val_evaluator': 'fair_val_evaluator',
    'test_evaluator': 'fair_test_evaluator',
    'train_cfg': 'fair_train_cfg',
    'val_cfg': 'fair_val_cfg',
    'test_cfg': 'fair_test_cfg',
    'optim_wrapper': 'fair_optim_wrapper',
    'param_scheduler': 'fair_param_scheduler',
    'default_hooks': 'fair_default_hooks',
    'randomness': 'fair_randomness',
    'auto_scale_lr': 'fair_auto_scale_lr',
    'custom_hooks': 'fair_custom_hooks',
    'visualizer': 'fair_visualizer',
    'log_processor': 'fair_log_processor',
    'log_level': 'fair_log_level',
    'env_cfg': 'fair_env_cfg',
}

CONTROLLED_KEYS = tuple(PROTOCOL_ASSIGNMENTS) + ('load_from', 'resume')


def experiment_paths():
    config_dir = Path(__file__).resolve().parent
    return sorted(
        path for path in config_dir.glob('*.py')
        if path.name != Path(__file__).name)


def static_audit(paths):
    failures = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        assignments = {
            node.targets[0].id: node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }

        for key, protocol_key in PROTOCOL_ASSIGNMENTS.items():
            expected = f'_base_.{protocol_key}'
            actual = assignments.get(key)
            if actual is None or ast.unparse(actual) != expected:
                failures.append(f'{path.name}: {key} must be {expected}')

        load_from = assignments.get('load_from')
        resume = assignments.get('resume')
        if load_from is None or ast.unparse(load_from) != 'None':
            failures.append(f'{path.name}: load_from must be None')
        if resume is None or ast.unparse(resume) != 'False':
            failures.append(f'{path.name}: resume must be False')

        model = assignments.get('model')
        if not isinstance(model, ast.Call):
            failures.append(f'{path.name}: model must be a dict')
            continue
        model_keywords = {item.arg: item.value for item in model.keywords}
        data_preprocessor = model_keywords.get('data_preprocessor')
        if (data_preprocessor is None
                or ast.unparse(data_preprocessor)
                != '_base_.fair_data_preprocessor'):
            failures.append(
                f'{path.name}: model.data_preprocessor must use fair protocol')

        if '_r50' in path.stem and 'dinov2' not in path.stem:
            backbone = model_keywords.get('backbone')
            if (backbone is None
                    or ast.unparse(backbone) != '_base_.fair_r50_backbone'):
                failures.append(
                    f'{path.name}: R50 backbone training settings differ')
        elif '_r101' in path.stem:
            backbone = model_keywords.get('backbone')
            if (backbone is None
                    or ast.unparse(backbone) != '_base_.fair_r101_backbone'):
                failures.append(
                    f'{path.name}: R101 backbone settings are not explicit')

    return failures


def normalized(value):
    value = deepcopy(value)
    if isinstance(value, dict):
        value.pop('_delete_', None)
        return {key: normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(item) for item in value]
    return value


def resolved_audit(paths):
    from mmengine.config import Config

    reference = Config.fromfile(paths[0])
    failures = []
    for path in paths[1:]:
        candidate = Config.fromfile(path)
        for key in CONTROLLED_KEYS:
            if normalized(candidate[key]) != normalized(reference[key]):
                failures.append(f'{path.name}: resolved {key} differs')
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--resolved',
        action='store_true',
        help='also load configs with MMEngine and compare resolved values')
    args = parser.parse_args()

    paths = experiment_paths()
    failures = static_audit(paths)
    if args.resolved:
        failures.extend(resolved_audit(paths))

    if failures:
        raise AssertionError(
            'Non-architectural settings differ:\n  ' + '\n  '.join(failures))

    mode = 'static and resolved' if args.resolved else 'static'
    print(f'PASS: {len(paths)} configs share the same protocol ({mode} audit).')


if __name__ == '__main__':
    main()
