#!/usr/bin/env python3
"""Run E063, then conditionally choose the E064/E065 final structure."""

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path('/root/autodl-tmp/outputs/HTL-ReID')
CHAIN_NAME = 'E063_E065_final_conditional'
BASE_MAP = 69.79
BASE_RANK1 = 72.37
MIN_GAIN = 0.50

E063 = {
    'experiment': 'E063',
    'row': 'A3-I-CSHS-RGBNT201',
    'config': 'configs/RGBNT201/fusion/a3_isolated_consensus_specific_clean.yml',
    'output_name': 'E063_A3_isolated_consensus_specific_clean_RGBNT201_seed1111',
    'base_config': 'configs/RGBNT201/paper/base.yml',
    'dataset': 'RGBNT201',
    'input_size': (256, 128),
    'batch_size': 64,
    'base_lr': 0.0001,
    'backbone_lr_factor': 0.1,
}

CROSS_DATASETS = (
    {
        'experiment': 'E064',
        'dataset': 'MSVR310',
        'base_config': 'configs/MSVR310/paper/base.yml',
        'input_size': (128, 256),
        'batch_size': 64,
        'base_lr': 0.00035,
        'backbone_lr_factor': 1.0,
        'output_name': 'E064_A3_final_MSVR310_seed1111',
        'candidate': {
            'row': 'A3-I-CSHS-MSVR310',
            'config': 'configs/MSVR310/fusion/a3_isolated_consensus_specific_clean.yml',
        },
        'fallback': {
            'row': 'A3-I-MSVR310',
            'config': 'configs/MSVR310/fusion/a3_isolated_clean.yml',
        },
    },
    {
        'experiment': 'E065',
        'dataset': 'RGBNT100',
        'base_config': 'configs/RGBNT100/paper/base.yml',
        'input_size': (128, 256),
        'batch_size': 128,
        'base_lr': 0.00035,
        'backbone_lr_factor': 0.8,
        'output_name': 'E065_A3_final_RGBNT100_seed1111',
        'candidate': {
            'row': 'A3-I-CSHS-RGBNT100',
            'config': 'configs/RGBNT100/fusion/a3_isolated_consensus_specific_clean.yml',
        },
        'fallback': {
            'row': 'A3-I-RGBNT100',
            'config': 'configs/RGBNT100/fusion/a3_isolated_clean.yml',
        },
    },
)


def child_command(python, spec, dry_run=False):
    command = [
        python,
        'tools/run_rgbnt201_fusion.py',
        '--single-experiment', spec['experiment'],
        '--single-row', spec['row'],
        '--single-config', spec['config'],
        '--single-output-name', spec['output_name'],
        '--base-config', spec['base_config'],
        '--dataset', spec['dataset'],
        '--input-size', *(str(value) for value in spec['input_size']),
        '--seed', '1111',
        '--eval-period', '50',
        '--expected-train-epochs', '50',
        '--expected-max-epochs', '50',
        '--expected-base-lr', str(spec['base_lr']),
        '--expected-batch-size', str(spec['batch_size']),
        '--expected-backbone-lr-factor', str(spec['backbone_lr_factor']),
        '--expected-warmup-iters', '10',
        '--expected-resume-path', '',
        '--expected-strict-determinism', '0',
    ]
    if dry_run:
        command.append('--dry-run')
    return command


def materialize_cross_spec(item, branch):
    spec = dict(item)
    spec.update(item[branch])
    spec.pop('candidate')
    spec.pop('fallback')
    return spec


def read_result(output_dir, spec, returncode, commit):
    result_path = output_dir / 'run_result.json'
    if result_path.exists():
        return json.loads(result_path.read_text(encoding='utf-8'))
    return {
        'experiment': spec['experiment'],
        'row': spec['row'],
        'status': 'runner_failed',
        'returncode': returncode,
        'config': spec['config'],
        'output_dir': str(output_dir),
        'commit': commit,
    }


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')


def gate_e063(result):
    try:
        map_value = float(result['mAP'])
        rank1 = float(result['Rank1'])
    except (KeyError, TypeError, ValueError):
        raise ValueError('E063 completed without numeric mAP/Rank1')
    passed = (
        (map_value >= BASE_MAP + MIN_GAIN and rank1 >= BASE_RANK1) or
        (rank1 >= BASE_RANK1 + MIN_GAIN and map_value >= BASE_MAP)
    )
    return {
        'baseline': {'mAP': BASE_MAP, 'Rank1': BASE_RANK1},
        'minimum_gain': MIN_GAIN,
        'e063': {'mAP': map_value, 'Rank1': rank1},
        'delta': {
            'mAP': round(map_value - BASE_MAP, 4),
            'Rank1': round(rank1 - BASE_RANK1, 4),
        },
        'passed': passed,
        'selected_branch': 'candidate' if passed else 'fallback',
        'rule': (
            '(mAP>=70.29 and Rank1>=72.37) or '
            '(Rank1>=72.87 and mAP>=69.79)'
        ),
    }


def run_one(args, spec, commit):
    command = child_command(args.python, spec)
    completed = subprocess.run(command, cwd=args.repo_root, check=False)
    output_dir = args.output_root / spec['output_name']
    return read_result(output_dir, spec, completed.returncode, commit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', default='/root/miniconda3/bin/python')
    parser.add_argument('--repo-root', type=Path, default=REPO_ROOT)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_root = args.output_root.resolve()

    if args.dry_run:
        candidates = [E063]
        for item in CROSS_DATASETS:
            candidates.extend([
                materialize_cross_spec(item, 'candidate'),
                materialize_cross_spec(item, 'fallback'),
            ])
        for spec in candidates:
            completed = subprocess.run(
                child_command(args.python, spec, dry_run=True),
                cwd=args.repo_root, check=False)
            if completed.returncode != 0:
                return completed.returncode
        print(json.dumps({
            'gate_baseline': {'mAP': BASE_MAP, 'Rank1': BASE_RANK1},
            'minimum_gain': MIN_GAIN,
            'candidate_on_pass': True,
            'fallback_on_metric_failure': True,
        }, indent=2, ensure_ascii=False))
        return 0

    chain_dir = args.output_root / CHAIN_NAME
    output_dirs = [args.output_root / E063['output_name']] + [
        args.output_root / item['output_name'] for item in CROSS_DATASETS]
    conflicts = [
        path for path in (chain_dir, *output_dirs) if path.exists()]
    if conflicts:
        raise FileExistsError(
            'refusing to overwrite: {}'.format(
                ', '.join(str(path) for path in conflicts)))

    chain_dir.mkdir(parents=True)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=args.repo_root, text=True).strip()
    plan = {
        'commit': commit,
        'e063': E063,
        'cross_datasets': CROSS_DATASETS,
        'gate': {
            'baseline_mAP': BASE_MAP,
            'baseline_Rank1': BASE_RANK1,
            'minimum_gain': MIN_GAIN,
        },
    }
    write_json(chain_dir / 'plan.json', plan)
    chain_dir.joinpath('RUNNING').write_text(
        dt.datetime.now().astimezone().isoformat() + '\n', encoding='utf-8')

    results = []
    result = run_one(args, E063, commit)
    results.append(result)
    write_json(chain_dir / 'summary.json', results)
    if result.get('status') != 'completed':
        chain_dir.joinpath('RUNNING').unlink(missing_ok=True)
        write_json(chain_dir / 'FAILED', {
            'failed_experiment': 'E063', 'results': results})
        return 1

    try:
        decision = gate_e063(result)
    except ValueError as error:
        chain_dir.joinpath('RUNNING').unlink(missing_ok=True)
        write_json(chain_dir / 'FAILED', {
            'failed_experiment': 'E063', 'error': str(error),
            'results': results})
        return 1
    write_json(chain_dir / 'decision.json', decision)

    branch = decision['selected_branch']
    for item in CROSS_DATASETS:
        spec = materialize_cross_spec(item, branch)
        result = run_one(args, spec, commit)
        results.append(result)
        write_json(chain_dir / 'summary.json', results)
        if result.get('status') != 'completed':
            chain_dir.joinpath('RUNNING').unlink(missing_ok=True)
            write_json(chain_dir / 'FAILED', {
                'failed_experiment': spec['experiment'],
                'decision': decision,
                'results': results,
            })
            return 1

    chain_dir.joinpath('RUNNING').unlink(missing_ok=True)
    write_json(chain_dir / 'DONE', {
        'decision': decision,
        'results': results,
    })
    return 0


if __name__ == '__main__':
    sys.exit(main())
