#!/usr/bin/env python3
"""Fine descriptor/TTA search plus E043 checkpoint soup and ensembling."""

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
from tools.sweep_descriptor_weights import extract_components
from utils.metrics import eval_func


COMPONENT_NAMES = ('aci', 'original', 'moe')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def float_grid(start, stop, step):
    count = int(round((stop - start) / step))
    return [round(start + index * step, 6) for index in range(count + 1)]


def local_grid(center, radius, step, lower, upper):
    start = max(lower, center - radius)
    stop = min(upper, center + radius)
    return float_grid(start, stop, step)


def blend_components(original, flipped, alpha):
    if alpha <= 0.0:
        return original
    if alpha >= 1.0:
        return flipped
    return {
        name: F.normalize(
            (1.0 - alpha) * original[name] + alpha * flipped[name],
            dim=1)
        for name in original
    }


def component_distances(components, num_query):
    distances = {}
    for name in COMPONENT_NAMES:
        features = components[name]
        query = features[:num_query].cuda(non_blocking=True)
        gallery = features[num_query:].cuda(non_blocking=True)
        distance = (2.0 - 2.0 * query @ gallery.t()).clamp_min_(0.0)
        distances[name] = distance.cpu().numpy()
        del query, gallery, distance
    return distances


def combine_distances(distances, cls_weight, moe_weight):
    return (
        distances['aci'] +
        (float(cls_weight) ** 2) * distances['original'] +
        (float(moe_weight) ** 2) * distances['moe']
    )


def metric(distance, pids, camids, num_query):
    cmc, mean_ap = eval_func(
        distance,
        pids[:num_query], pids[num_query:],
        camids[:num_query], camids[num_query:],
        max_rank=50)
    return {
        'mAP': round(100.0 * float(mean_ap), 4),
        'Rank1': round(100.0 * float(cmc[0]), 4),
        'Rank5': round(100.0 * float(cmc[4]), 4),
        'Rank10': round(100.0 * float(cmc[9]), 4),
    }


def select_best(results, target_map, key):
    feasible = [item for item in results if item['mAP'] >= target_map]
    return max(feasible or results, key=key)


def search(model_name, original, flipped, pids, camids, num_query,
           alpha_values, cls_values, moe_values, stage, target_map,
           target_rank1, seen=None):
    results = []
    seen = seen if seen is not None else set()
    for alpha in alpha_values:
        blended = blend_components(original, flipped, alpha)
        distances = component_distances(blended, num_query)
        for cls_weight, moe_weight in itertools.product(cls_values, moe_values):
            identity = (
                model_name, round(float(alpha), 6),
                round(float(cls_weight), 6), round(float(moe_weight), 6))
            if identity in seen:
                continue
            seen.add(identity)
            result = {
                'kind': 'single',
                'model': model_name,
                'stage': stage,
                'flip_alpha': float(alpha),
                'weights': {
                    'aci': 1.0,
                    'original': float(cls_weight),
                    'part': 0.0,
                    'moe': float(moe_weight),
                },
                **metric(
                    combine_distances(distances, cls_weight, moe_weight),
                    pids, camids, num_query),
            }
            result['target_hit'] = (
                result['mAP'] >= target_map and
                result['Rank1'] >= target_rank1)
            results.append(result)
    return results


def candidate_distance(candidate, cache, num_query):
    original, flipped = cache[candidate['model']]
    components = blend_components(
        original, flipped, candidate['flip_alpha'])
    distances = component_distances(components, num_query)
    return combine_distances(
        distances, candidate['weights']['original'],
        candidate['weights']['moe'])


def interpolate_states(first, second, alpha):
    if tuple(first.keys()) != tuple(second.keys()):
        raise ValueError('checkpoint state-dict keys differ')
    mixed = {}
    for key in first:
        left, right = first[key], second[key]
        if left.shape != right.shape:
            raise ValueError('checkpoint shape mismatch for {}'.format(key))
        if torch.is_floating_point(left):
            mixed[key] = torch.lerp(left, right, float(alpha))
        else:
            mixed[key] = right.clone() if alpha >= 0.5 else left.clone()
    return mixed


def extract_model(model, val_loader, checkpoint=None, state_dict=None):
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    else:
        model.load_param(str(checkpoint))
    original, flipped, pids, camids = extract_components(
        model, val_loader, use_flip=True)
    original = {name: original[name] for name in COMPONENT_NAMES}
    flipped = {name: flipped[name] for name in COMPONENT_NAMES}
    return original, flipped, pids, camids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', action='append', required=True)
    parser.add_argument('--rank1-checkpoint', type=Path, required=True)
    parser.add_argument('--map-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--target-map', type=float, default=73.7)
    parser.add_argument('--target-rank1', type=float, default=80.5)
    args = parser.parse_args()

    for config_file in args.config_file:
        cfg.merge_from_file(config_file)
    if str(cfg.TEST.RE_RANKING).lower() != 'no':
        raise ValueError('E045 requires re-ranking disabled')
    for checkpoint in (args.rank1_checkpoint, args.map_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
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
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num).cuda()
    model.eval()

    cache = {}
    pids = camids = None
    for name, checkpoint in (
            ('epoch31_rank1', args.rank1_checkpoint),
            ('epoch33_map', args.map_checkpoint)):
        original, flipped, model_pids, model_camids = extract_model(
            model, val_loader, checkpoint=checkpoint)
        cache[name] = (original, flipped)
        if pids is None:
            pids, camids = model_pids, model_camids
        elif (not np.array_equal(pids, model_pids) or
              not np.array_equal(camids, model_camids)):
            raise RuntimeError('validation order changed between checkpoints')

    if len(pids) != cache['epoch31_rank1'][0]['aci'].size(0):
        raise RuntimeError('feature and label counts differ')

    coarse_alpha = (0.0, 0.25, 0.5, 0.75, 1.0)
    coarse_cls = float_grid(0.2, 1.0, 0.1)
    coarse_moe = float_grid(0.5, 1.5, 0.1)
    all_results = []
    seen_by_model = {}

    def staged_search(model_name):
        seen = seen_by_model.setdefault(model_name, set())
        original, flipped = cache[model_name]
        coarse = search(
            model_name, original, flipped, pids, camids, num_query,
            coarse_alpha, coarse_cls, coarse_moe, 'coarse',
            args.target_map, args.target_rank1, seen)
        all_results.extend(coarse)
        coarse = [
            item for item in all_results
            if item['kind'] == 'single' and item['model'] == model_name and
            item['stage'] == 'coarse']
        centers = {
            'rank1': select_best(
                coarse, args.target_map,
                lambda item: (item['Rank1'], item['mAP'])),
            'map': max(coarse, key=lambda item: (item['mAP'], item['Rank1'])),
        }
        for objective, center in centers.items():
            refined = search(
                model_name, original, flipped, pids, camids, num_query,
                local_grid(center['flip_alpha'], 0.2, 0.05, 0.0, 1.0),
                local_grid(center['weights']['original'], 0.1, 0.02, 0.0, 2.0),
                local_grid(center['weights']['moe'], 0.1, 0.02, 0.0, 2.0),
                'refine_{}'.format(objective), args.target_map,
                args.target_rank1, seen)
            all_results.extend(refined)

    staged_search('epoch31_rank1')
    staged_search('epoch33_map')

    first_state = torch.load(
        args.rank1_checkpoint, map_location='cpu', weights_only=True)
    second_state = torch.load(
        args.map_checkpoint, map_location='cpu', weights_only=True)
    soup_names = []
    for alpha in (0.25, 0.5, 0.75):
        name = 'soup_{:.2f}'.format(alpha)
        state = interpolate_states(first_state, second_state, alpha)
        original, flipped, model_pids, model_camids = extract_model(
            model, val_loader, state_dict=state)
        if (not np.array_equal(pids, model_pids) or
                not np.array_equal(camids, model_camids)):
            raise RuntimeError('validation order changed for checkpoint soup')
        cache[name] = (original, flipped)
        soup_names.append(name)
        seen = seen_by_model.setdefault(name, set())
        all_results.extend(search(
            name, original, flipped, pids, camids, num_query,
            coarse_alpha, coarse_cls, coarse_moe, 'coarse',
            args.target_map, args.target_rank1, seen))

    best_soup_name = max(
        soup_names,
        key=lambda name: select_best(
            [item for item in all_results if item['model'] == name],
            args.target_map,
            lambda item: (item['Rank1'], item['mAP']))['Rank1'])
    staged_search(best_soup_name)

    best_per_model = []
    for name in cache:
        model_results = [
            item for item in all_results if item['model'] == name]
        best_per_model.append(select_best(
            model_results, args.target_map,
            lambda item: (item['Rank1'], item['mAP'])))

    ensemble_results = []
    ensemble_distances = {
        item['model']: candidate_distance(item, cache, num_query)
        for item in best_per_model
    }
    for left, right in itertools.combinations(best_per_model, 2):
        left_distance = ensemble_distances[left['model']]
        right_distance = ensemble_distances[right['model']]
        for alpha in float_grid(0.0, 1.0, 0.025):
            distance = (1.0 - alpha) * left_distance + alpha * right_distance
            result = {
                'kind': 'distance_ensemble',
                'models': [left['model'], right['model']],
                'ensemble_alpha': float(alpha),
                'left_candidate': left,
                'right_candidate': right,
                **metric(distance, pids, camids, num_query),
            }
            result['target_hit'] = (
                result['mAP'] >= args.target_map and
                result['Rank1'] >= args.target_rank1)
            ensemble_results.append(result)

    all_results.extend(ensemble_results)
    target_hits = [item for item in all_results if item['target_hit']]
    best_rank1 = select_best(
        all_results, args.target_map,
        lambda item: (item['Rank1'], item['mAP']))
    best_map = max(
        all_results, key=lambda item: (item['mAP'], item['Rank1']))

    best_single = select_best(
        [item for item in all_results if item['kind'] == 'single'],
        args.target_map, lambda item: (item['Rank1'], item['mAP']))
    if best_single['model'].startswith('soup_'):
        soup_alpha = float(best_single['model'].split('_', 1)[1])
        torch.save(
            interpolate_states(first_state, second_state, soup_alpha),
            args.output_dir / 'HTL-ReID_best_soup.pth')

    payload = {
        'experiment': 'E045',
        'status': 'completed',
        'commit': commit,
        'checkpoints': {
            'epoch31_rank1': str(args.rank1_checkpoint),
            'epoch33_map': str(args.map_checkpoint),
        },
        're_ranking': 'no',
        'elapsed_seconds': round(time.monotonic() - started, 1),
        'num_candidates': len(all_results),
        'num_target_hits': len(target_hits),
        'target': {'mAP': args.target_map, 'Rank1': args.target_rank1},
        'best_rank1': best_rank1,
        'best_mAP': best_map,
        'best_single_model': best_single,
        'best_per_model': best_per_model,
        'results': sorted(
            all_results,
            key=lambda item: (item['Rank1'], item['mAP']),
            reverse=True),
    }
    (args.output_dir / 'result.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    (args.output_dir / 'DONE').write_text(
        json.dumps({
            key: payload[key]
            for key in (
                'experiment', 'status', 'elapsed_seconds',
                'num_candidates', 'num_target_hits', 'best_rank1',
                'best_mAP', 'best_single_model')
        }, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(best_rank1, ensure_ascii=False))
    print(json.dumps(best_map, ensure_ascii=False))
    print(json.dumps(best_single, ensure_ascii=False))
    print('CANDIDATES={} TARGET_HITS={}'.format(
        len(all_results), len(target_hits)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
