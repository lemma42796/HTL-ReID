#!/usr/bin/env python3
"""Run post-final efficiency, robustness, and visualization evidence."""

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path('/root/autodl-tmp/outputs/HTL-ReID')
UPSTREAM_CHAIN = 'E063_E065_final_conditional'
CHAIN_NAME = 'E066_E068_review_evidence'
BASE_CONFIG = 'configs/RGBNT201/paper/base.yml'
BASELINE_CONFIG = 'configs/RGBNT201/ablations/t14_chain/a0_backbone.yml'
OLD_T14_CONFIG = 'configs/RGBNT201/fusion/t14_clean_baseline.yml'
BASELINE_CHECKPOINT = (
    '/root/autodl-tmp/outputs/HTL-ReID/'
    'E049_A0_backbone_RGBNT201_seed1111/HTL-ReID_best.pth')
OLD_T14_CHECKPOINT = (
    '/root/autodl-tmp/outputs/HTL-ReID/'
    'E057_T14_clean_baseline_RGBNT201_seed1111/HTL-ReID_best.pth')
FINAL_BRANCHES = {
    'candidate': {
        'config': (
            'configs/RGBNT201/fusion/'
            'a3_isolated_consensus_specific_clean.yml'),
        'checkpoint': (
            '/root/autodl-tmp/outputs/HTL-ReID/'
            'E063_A3_isolated_consensus_specific_clean_'
            'RGBNT201_seed1111/HTL-ReID_best.pth'),
    },
    'fallback': {
        'config': 'configs/RGBNT201/fusion/a3_isolated_clean.yml',
        'checkpoint': (
            '/root/autodl-tmp/outputs/HTL-ReID/'
            'E060_A3_isolated_clean_RGBNT201_seed1111/HTL-ReID_best.pth'),
    },
}


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')


def run_task(repo_root, timeout_bin, task, chain_dir):
    command = [
        timeout_bin, '--signal=TERM', '--kill-after=10s', task['timeout'],
        *task['command'],
    ]
    log_path = chain_dir / '{}.log'.format(task['experiment'])
    with log_path.open('w', encoding='utf-8') as handle:
        handle.write(' '.join(command) + '\n\n')
        handle.flush()
        completed = subprocess.run(
            command, cwd=repo_root, stdout=handle,
            stderr=subprocess.STDOUT, check=False)
    return {
        'experiment': task['experiment'],
        'name': task['name'],
        'returncode': completed.returncode,
        'status': 'completed' if completed.returncode == 0 else (
            'timeout' if completed.returncode == 124 else 'failed'),
        'output_dir': task['output_dir'],
        'command': command,
        'log': str(log_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', default='/root/miniconda3/bin/python')
    parser.add_argument('--repo-root', type=Path, default=REPO_ROOT)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument(
        '--selected-branch', choices=tuple(FINAL_BRANCHES),
        help='dry-run only: validate one branch before upstream completion')
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_root = args.output_root.resolve()

    if args.selected_branch is not None and not args.dry_run:
        raise ValueError('--selected-branch is allowed only with --dry-run')
    upstream_dir = args.output_root / UPSTREAM_CHAIN
    if args.selected_branch is not None:
        branch = args.selected_branch
        decision = {
            'selected_branch': branch,
            'dry_run_override': True,
        }
    else:
        if not (upstream_dir / 'DONE').is_file():
            raise RuntimeError('E063-E065 must complete before review evidence')
        decision_path = upstream_dir / 'decision.json'
        if not decision_path.is_file():
            raise FileNotFoundError(decision_path)
        decision = json.loads(decision_path.read_text(encoding='utf-8'))
        branch = decision.get('selected_branch')
    if branch not in FINAL_BRANCHES:
        raise ValueError('unknown final branch: {!r}'.format(branch))
    final = FINAL_BRANCHES[branch]

    output_dirs = {
        'E066': args.output_root / 'E066_final_efficiency',
        'E067': args.output_root / 'E067_final_robustness',
        'E068': args.output_root / 'E068_final_visual_analysis',
    }
    tasks = [
        {
            'experiment': 'E066',
            'name': 'final efficiency and complexity',
            'timeout': '15m',
            'output_dir': str(output_dirs['E066']),
            'command': [
                args.python, 'tools/measure_final_efficiency.py',
                '--base-config', BASE_CONFIG,
                '--baseline-config', BASELINE_CONFIG,
                '--final-config', final['config'],
                '--output-dir', str(output_dirs['E066']),
                '--seed', '1111', '--batch-size', '16',
                '--warmup', '20', '--repeats', '50',
            ],
        },
        {
            'experiment': 'E067',
            'name': 'missing modality and controlled degradation',
            'timeout': '20m',
            'output_dir': str(output_dirs['E067']),
            'command': [
                args.python, 'tools/evaluate_final_robustness.py',
                '--base-config', BASE_CONFIG,
                '--baseline-config', OLD_T14_CONFIG,
                '--baseline-checkpoint', OLD_T14_CHECKPOINT,
                '--final-config', final['config'],
                '--final-checkpoint', final['checkpoint'],
                '--output-dir', str(output_dirs['E067']),
                '--seed', '1111', '--batch-size', '64',
            ],
        },
        {
            'experiment': 'E068',
            'name': 'clean final visual analysis',
            'timeout': '15m',
            'output_dir': str(output_dirs['E068']),
            'command': [
                args.python, 'tools/generate_final_review_visualizations.py',
                '--base-config', BASE_CONFIG,
                '--baseline-config', BASELINE_CONFIG,
                '--baseline-checkpoint', BASELINE_CHECKPOINT,
                '--final-config', final['config'],
                '--final-checkpoint', final['checkpoint'],
                '--robustness-summary',
                str(output_dirs['E067'] / 'summary.json'),
                '--output-dir', str(output_dirs['E068']),
                '--seed', '1111', '--batch-size', '64',
            ],
        },
    ]

    required = [
        args.repo_root / BASE_CONFIG,
        args.repo_root / BASELINE_CONFIG,
        args.repo_root / OLD_T14_CONFIG,
        args.repo_root / final['config'],
        Path(BASELINE_CHECKPOINT), Path(OLD_T14_CHECKPOINT),
        Path(final['checkpoint']),
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            'missing required inputs: {}'.format(
                ', '.join(str(path) for path in missing)))

    plan = {
        'upstream_decision': decision,
        'selected_branch': branch,
        'final': final,
        'tasks': tasks,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    chain_dir = args.output_root / CHAIN_NAME
    conflicts = [
        path for path in (chain_dir, *output_dirs.values()) if path.exists()]
    if conflicts:
        raise FileExistsError(
            'refusing to overwrite: {}'.format(
                ', '.join(str(path) for path in conflicts)))
    if shutil.disk_usage(args.output_root).free < 2 * 1024 ** 3:
        raise RuntimeError('less than 2 GiB free under output root')
    timeout_bin = shutil.which('timeout')
    if timeout_bin is None:
        raise RuntimeError('GNU timeout is required')

    chain_dir.mkdir(parents=True)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=args.repo_root, text=True).strip()
    plan['commit'] = commit
    write_json(chain_dir / 'plan.json', plan)
    chain_dir.joinpath('RUNNING').write_text(
        dt.datetime.now().astimezone().isoformat() + '\n', encoding='utf-8')
    results = []
    for task in tasks:
        result = run_task(args.repo_root, timeout_bin, task, chain_dir)
        results.append(result)
        write_json(chain_dir / 'summary.json', results)
        if result['status'] != 'completed':
            chain_dir.joinpath('RUNNING').unlink(missing_ok=True)
            write_json(chain_dir / 'FAILED', {
                'selected_branch': branch,
                'failed_experiment': task['experiment'],
                'results': results,
            })
            return 1
    chain_dir.joinpath('RUNNING').unlink(missing_ok=True)
    write_json(chain_dir / 'DONE', {
        'selected_branch': branch,
        'results': results,
    })
    return 0


if __name__ == '__main__':
    sys.exit(main())
