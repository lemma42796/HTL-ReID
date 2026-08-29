#!/usr/bin/env python3
"""Run the missing seeds for the clean end-to-end/isolation ACI pair."""

import argparse
import datetime as dt
import json
import statistics
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path('/root/autodl-tmp/outputs/HTL-ReID')
CHAIN_NAME = 'E073_E076_isolation_multiseed_pair'
ROWS = (
    ('E073', 'A2-ACI-END2END-S2222',
     'configs/RGBNT201/ablations/t14_chain/a2_facr.yml',
     'E073_A2_aci_end2end_RGBNT201_seed2222', 2222),
    ('E074', 'A2-ACI-ISOLATED-S2222',
     'configs/RGBNT201/ablations/t14_chain/a2_facr_isolated.yml',
     'E074_A2_aci_isolated_RGBNT201_seed2222', 2222),
    ('E075', 'A2-ACI-END2END-S3333',
     'configs/RGBNT201/ablations/t14_chain/a2_facr.yml',
     'E075_A2_aci_end2end_RGBNT201_seed3333', 3333),
    ('E076', 'A2-ACI-ISOLATED-S3333',
     'configs/RGBNT201/ablations/t14_chain/a2_facr_isolated.yml',
     'E076_A2_aci_isolated_RGBNT201_seed3333', 3333),
)
SEED1111 = {
    'end_to_end': OUTPUT_ROOT / 'E061_A2_facr_clean_RGBNT201_seed1111',
    'isolated': OUTPUT_ROOT / 'E062_A2_facr_isolated_clean_RGBNT201_seed1111',
}


def child_command(python, experiment, row, config, output_name, seed,
                  dry_run=False):
    command = [
        python,
        'tools/run_rgbnt201_fusion.py',
        '--single-experiment', experiment,
        '--single-row', row,
        '--single-config', config,
        '--single-output-name', output_name,
        '--seed', str(seed),
        '--eval-period', '50',
        '--expected-train-epochs', '50',
        '--expected-max-epochs', '50',
        '--expected-base-lr', '0.0001',
        '--expected-batch-size', '64',
        '--expected-backbone-lr-factor', '0.1',
        '--expected-warmup-iters', '10',
        '--expected-resume-path', '',
        '--expected-strict-determinism', '0',
    ]
    if dry_run:
        command.append('--dry-run')
    return command


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_result(output_dir, experiment, row, config, seed, returncode, commit):
    result_path = output_dir / 'run_result.json'
    if result_path.exists():
        result = read_json(result_path)
        result['seed'] = seed
        return result
    return {
        'experiment': experiment,
        'row': row,
        'seed': seed,
        'status': 'runner_failed',
        'returncode': returncode,
        'config': config,
        'output_dir': str(output_dir),
        'commit': commit,
    }


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')


def summarize_three_seeds(results):
    existing = {
        1111: {
            'end_to_end': read_json(SEED1111['end_to_end'] / 'run_result.json'),
            'isolated': read_json(SEED1111['isolated'] / 'run_result.json'),
        }
    }
    by_experiment = {item['experiment']: item for item in results}
    existing[2222] = {
        'end_to_end': by_experiment['E073'],
        'isolated': by_experiment['E074'],
    }
    existing[3333] = {
        'end_to_end': by_experiment['E075'],
        'isolated': by_experiment['E076'],
    }
    metrics = ('mAP', 'Rank1', 'Rank5', 'Rank10')
    summary = {'seeds': existing, 'aggregate': {}, 'paired_delta': {}}
    for variant in ('end_to_end', 'isolated'):
        summary['aggregate'][variant] = {}
        for metric in metrics:
            values = [float(existing[seed][variant][metric]) for seed in sorted(existing)]
            summary['aggregate'][variant][metric] = {
                'values': values,
                'mean': statistics.mean(values),
                'sample_std': statistics.stdev(values),
            }
    for metric in metrics:
        deltas = [
            float(existing[seed]['isolated'][metric]) -
            float(existing[seed]['end_to_end'][metric])
            for seed in sorted(existing)
        ]
        summary['paired_delta'][metric] = {
            'values': deltas,
            'mean': statistics.mean(deltas),
            'sample_std': statistics.stdev(deltas),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--python', default='/root/miniconda3/bin/python')
    parser.add_argument('--repo-root', type=Path, default=REPO_ROOT)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_root = args.output_root.resolve()

    commands = [
        child_command(
            args.python, experiment, row, config, output_name, seed,
            dry_run=args.dry_run)
        for experiment, row, config, output_name, seed in ROWS
    ]
    if args.dry_run:
        for command in commands:
            completed = subprocess.run(command, cwd=args.repo_root, check=False)
            if completed.returncode != 0:
                return completed.returncode
        return 0

    chain_dir = args.output_root / CHAIN_NAME
    output_dirs = [args.output_root / item[3] for item in ROWS]
    conflicts = [path for path in (chain_dir, *output_dirs) if path.exists()]
    if conflicts:
        raise FileExistsError(
            'refusing to overwrite: {}'.format(
                ', '.join(str(path) for path in conflicts)))

    for path in SEED1111.values():
        if not path.joinpath('run_result.json').exists():
            raise FileNotFoundError(
                'missing seed-1111 paired result: {}'.format(path))

    chain_dir.mkdir(parents=True)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=args.repo_root, text=True).strip()
    plan = []
    for item, command in zip(ROWS, commands):
        experiment, row, config, output_name, seed = item
        plan.append({
            'experiment': experiment,
            'row': row,
            'seed': seed,
            'config': config,
            'output_dir': str(args.output_root / output_name),
            'command': command,
        })
    write_json(chain_dir / 'plan.json', plan)
    chain_dir.joinpath('commit.txt').write_text(commit + '\n', encoding='utf-8')
    chain_dir.joinpath('RUNNING').write_text(
        dt.datetime.now().astimezone().isoformat() + '\n', encoding='utf-8')

    results = []
    for index, (item, command, output_dir) in enumerate(
            zip(ROWS, commands, output_dirs)):
        experiment, row, config, _, seed = item
        completed = subprocess.run(command, cwd=args.repo_root, check=False)
        result = read_result(
            output_dir, experiment, row, config, seed,
            completed.returncode, commit)
        results.append(result)
        write_json(chain_dir / 'summary.json', results)
        if result.get('status') != 'completed':
            chain_dir.joinpath('RUNNING').unlink(missing_ok=True)
            write_json(chain_dir / 'FAILED', {
                'failed_index': index,
                'failed_experiment': experiment,
                'results': results,
            })
            return 1

    write_json(chain_dir / 'three_seed_summary.json',
               summarize_three_seeds(results))
    chain_dir.joinpath('RUNNING').unlink(missing_ok=True)
    write_json(chain_dir / 'DONE', results)
    return 0


if __name__ == '__main__':
    sys.exit(main())
