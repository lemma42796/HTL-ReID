#!/usr/bin/env python3
"""Cache E043 descriptor blocks once and sweep weighted retrieval distances."""

import argparse
import itertools
import json
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from data import make_dataloader
from modeling import make_model
from utils.metrics import eval_func


COMPONENT_NAMES = ('facr', 'original', 'part', 'moe')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalized_cpu(components):
    missing = [name for name in COMPONENT_NAMES if name not in components]
    if missing:
        raise RuntimeError('missing descriptor components: {}'.format(missing))
    return {
        name: F.normalize(components[name].float(), dim=1).cpu()
        for name in COMPONENT_NAMES
    }


def extract_components(model, val_loader, use_flip):
    original = {name: [] for name in COMPONENT_NAMES}
    flipped = {name: [] for name in COMPONENT_NAMES} if use_flip else None
    pids, camids = [], []
    model.eval()
    with torch.inference_mode():
        for images, vid, camid, camera_labels, view_labels, paths in val_loader:
            images = {
                name: tensor.cuda(non_blocking=True)
                for name, tensor in images.items()
            }
            camera_labels = camera_labels.cuda(non_blocking=True)
            view_labels = view_labels.cuda(non_blocking=True)
            blocks = model(
                images, cam_label=camera_labels, view_label=view_labels,
                mode=1, img_path=paths, return_descriptor_components=True)
            blocks = normalized_cpu(blocks)
            for name in COMPONENT_NAMES:
                original[name].append(blocks[name])

            if use_flip:
                flip_images = {
                    name: torch.flip(tensor, dims=(3,))
                    for name, tensor in images.items()
                }
                flip_blocks = model(
                    flip_images, cam_label=camera_labels,
                    view_label=view_labels, mode=1, img_path=paths,
                    return_descriptor_components=True)
                flip_blocks = normalized_cpu(flip_blocks)
                for name in COMPONENT_NAMES:
                    flipped[name].append(flip_blocks[name])

            pids.extend(np.asarray(vid).tolist())
            camids.extend(np.asarray(camid).tolist())

    original = {
        name: torch.cat(values, dim=0)
        for name, values in original.items()
    }
    if use_flip:
        flipped = {
            name: torch.cat(values, dim=0)
            for name, values in flipped.items()
        }
    return original, flipped, np.asarray(pids), np.asarray(camids)


def tta_average(original, flipped):
    return {
        name: F.normalize(original[name] + flipped[name], dim=1)
        for name in COMPONENT_NAMES
    }


def component_distances(components, num_query):
    distances = {}
    for name, features in components.items():
        query = features[:num_query].cuda(non_blocking=True)
        gallery = features[num_query:].cuda(non_blocking=True)
        distance = (2.0 - 2.0 * query @ gallery.t()).clamp_min_(0.0)
        distances[name] = distance.cpu().numpy()
        del query, gallery, distance
    return distances


def evaluate_grid(distances, pids, camids, num_query, weight_grid,
                  variant, target_map, target_rank1):
    q_pids, g_pids = pids[:num_query], pids[num_query:]
    q_camids, g_camids = camids[:num_query], camids[num_query:]
    results = []
    for original_weight, part_weight, moe_weight in itertools.product(
            weight_grid['original'], weight_grid['part'], weight_grid['moe']):
        weights = {
            'facr': 1.0,
            'original': float(original_weight),
            'part': float(part_weight),
            'moe': float(moe_weight),
        }
        distance = sum(
            (weight ** 2) * distances[name]
            for name, weight in weights.items()
        )
        cmc, mean_ap = eval_func(
            distance, q_pids, g_pids, q_camids, g_camids, max_rank=50)
        result = {
            'variant': variant,
            'weights': weights,
            'mAP': round(100.0 * float(mean_ap), 4),
            'Rank1': round(100.0 * float(cmc[0]), 4),
            'Rank5': round(100.0 * float(cmc[4]), 4),
            'Rank10': round(100.0 * float(cmc[9]), 4),
        }
        result['target_hit'] = (
            result['mAP'] >= target_map and result['Rank1'] >= target_rank1)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', action='append', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--original-weights', type=float, nargs='+',
                        default=(0.25, 0.5, 0.75))
    parser.add_argument('--part-weights', type=float, nargs='+',
                        default=(0.0, 0.15, 0.3))
    parser.add_argument('--moe-weights', type=float, nargs='+',
                        default=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument('--tta-flip', action='store_true')
    parser.add_argument('--target-map', type=float, default=73.7)
    parser.add_argument('--target-rank1', type=float, default=80.5)
    args = parser.parse_args()

    for config_file in args.config_file:
        cfg.merge_from_file(config_file)
    if str(cfg.TEST.RE_RANKING).lower() != 'no':
        raise ValueError('descriptor sweep requires re-ranking disabled')
    if not bool(cfg.MODEL.DECOUPLED_MOE) or not bool(cfg.MODEL.PART_BRANCH):
        raise ValueError('descriptor sweep requires the E043 MoE and Part branches')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU fallback is disabled')

    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.output_dir.joinpath('resolved_config.yml').write_text(
        cfg.dump(), encoding='utf-8')
    args.output_dir.joinpath('command.txt').write_text(
        shlex.join(sys.argv) + '\n', encoding='utf-8')
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
    args.output_dir.joinpath('commit.txt').write_text(
        commit + '\n', encoding='utf-8')

    set_seed(int(cfg.SOLVER.SEED))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    started = time.monotonic()
    loaders = make_dataloader(cfg)
    train_loader, train_loader_normal, val_loader = loaders[:3]
    num_query, num_classes, camera_num = loaders[3:6]
    del train_loader, train_loader_normal
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num)
    model.load_param(str(args.checkpoint))
    model.cuda()

    original, flipped, pids, camids = extract_components(
        model, val_loader, use_flip=args.tta_flip)
    if len(pids) != original['facr'].size(0):
        raise RuntimeError('feature and label counts differ')
    del model
    torch.cuda.empty_cache()

    weight_grid = {
        'original': [float(value) for value in args.original_weights],
        'part': [float(value) for value in args.part_weights],
        'moe': [float(value) for value in args.moe_weights],
    }
    results = evaluate_grid(
        component_distances(original, num_query), pids, camids, num_query,
        weight_grid, 'original', args.target_map, args.target_rank1)
    if args.tta_flip:
        averaged = tta_average(original, flipped)
        results.extend(evaluate_grid(
            component_distances(averaged, num_query), pids, camids,
            num_query, weight_grid, 'flip_tta', args.target_map,
            args.target_rank1))

    feasible = [item for item in results if item['mAP'] >= args.target_map]
    target_hits = [item for item in results if item['target_hit']]
    best_rank1 = max(feasible or results,
                     key=lambda item: (item['Rank1'], item['mAP']))
    best_map = max(results, key=lambda item: (item['mAP'], item['Rank1']))
    payload = {
        'experiment': 'E044',
        'status': 'completed',
        'checkpoint': str(args.checkpoint),
        'commit': commit,
        're_ranking': 'no',
        'num_query': int(num_query),
        'num_images': int(len(pids)),
        'grid': weight_grid,
        'variants': ['original'] + (['flip_tta'] if args.tta_flip else []),
        'elapsed_seconds': round(time.monotonic() - started, 1),
        'target': {'mAP': args.target_map, 'Rank1': args.target_rank1},
        'target_hit': bool(target_hits),
        'best_target_hit': max(
            target_hits, key=lambda item: (item['Rank1'], item['mAP']))
            if target_hits else None,
        'best_rank1_with_target_map': best_rank1,
        'best_mAP': best_map,
        'results': sorted(
            results, key=lambda item: (item['Rank1'], item['mAP']),
            reverse=True),
    }
    result_path = args.output_dir / 'result.json'
    result_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    args.output_dir.joinpath('DONE').write_text(
        json.dumps({
            'experiment': 'E044',
            'status': 'completed',
            'target_hit': bool(target_hits),
            'best_target_hit': payload['best_target_hit'],
            'best_rank1_with_target_map': best_rank1,
            'best_mAP': best_map,
            'elapsed_seconds': payload['elapsed_seconds'],
        }, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(payload['best_rank1_with_target_map'], ensure_ascii=False))
    print(json.dumps(payload['best_mAP'], ensure_ascii=False))
    print('TARGET_HITS={}'.format(len(target_hits)))
    print('saved {}'.format(result_path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
