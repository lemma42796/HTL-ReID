#!/usr/bin/env python3
"""Run the clean legacy baseline and CRUF sequentially on RGBNT201."""

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path('/root/autodl-tmp/outputs/HTL-ReID')
CHAIN_NAME = 'E057_E058_clean_CRUF'
ROWS = (
    (
        'E057',
        'T14-CLEAN-BASELINE',
        'configs/RGBNT201/fusion/t14_clean_baseline.yml',
        'E057_T14_clean_baseline_RGBNT201_seed1111',
    ),
    (
        'E058',
        'T15-CRUF-CLEAN',
        'configs/RGBNT201/fusion/t15_cruf_clean.yml',
        'E058_T15_CRUF_RGBNT201_seed1111',
    ),
)


def child_command(python, experiment, row, config, output_name, dry_run=False):
    command = [
        python,
        'tools/run_rgbnt201_fusion.py',
        '--single-experiment', experiment,
        '--single-row', row,
        '--single-config', config,
        '--single-output-name', output_name,
        '--seed', '1111',
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


def read_result(output_dir, experiment, row, config, returncode, commit):
    result_path = output_dir / 'run_result.json'
    if result_path.exists():
        return json.loads(result_path.read_text(encoding='utf-8'))
    return {
        'experiment': experiment,
        'row': row,
        'status': 'runner_failed',
        'returncode': returncode,
        'config': config,
        'output_dir': str(output_dir),
        'commit': commit,
    }


def write_summary(chain_dir, results):
    chain_dir.joinpath('summary.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')


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
            args.python, experiment, row, config, output_name,
            dry_run=args.dry_run)
        for experiment, row, config, output_name in ROWS
    ]
    if args.dry_run:
        for command in commands:
            completed = subprocess.run(
                command, cwd=args.repo_root, check=False)
            if completed.returncode != 0:
                return completed.returncode
        return 0

    chain_dir = args.output_root / CHAIN_NAME
    output_dirs = [args.output_root / item[3] for item in ROWS]
    conflicts = [
        path for path in (chain_dir, *output_dirs) if path.exists()]
    if conflicts:
        raise FileExistsError(
            'refusing to overwrite: {}'.format(
                ', '.join(str(path) for path in conflicts)))

    chain_dir.mkdir(parents=True)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=args.repo_root, text=True).strip()
    plan = []
    for (experiment, row, config, output_name), command in zip(ROWS, commands):
        plan.append({
            'experiment': experiment,
            'row': row,
            'config': config,
            'output_dir': str(args.output_root / output_name),
            'command': command,
        })
    chain_dir.joinpath('plan.json').write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    chain_dir.joinpath('commit.txt').write_text(commit + '\n', encoding='utf-8')
    chain_dir.joinpath('RUNNING').write_text(
        dt.datetime.now().astimezone().isoformat() + '\n', encoding='utf-8')

    results = []
    for index, ((experiment, row, config, _), command, output_dir) in enumerate(
            zip(ROWS, commands, output_dirs)):
        completed = subprocess.run(
            command, cwd=args.repo_root, check=False)
        result = read_result(
            output_dir, experiment, row, config,
            completed.returncode, commit)
        results.append(result)
        write_summary(chain_dir, results)
        if result.get('status') != 'completed':
            chain_dir.joinpath('RUNNING').unlink(missing_ok=True)
            chain_dir.joinpath('FAILED').write_text(
                json.dumps({
                    'failed_index': index,
                    'failed_experiment': experiment,
                    'results': results,
                }, indent=2, ensure_ascii=False) + '\n',
                encoding='utf-8')
            return 1

    chain_dir.joinpath('RUNNING').unlink(missing_ok=True)
    chain_dir.joinpath('DONE').write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
