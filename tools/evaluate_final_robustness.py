#!/usr/bin/env python3
"""Evaluate missing-modality and controlled-occlusion robustness."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from review_evidence_common import build_eval_run, require_cuda, set_seed
from utils.metrics import R1_mAP_eval


MODALITY_KEYS = {'RGB': 'RGB', 'NIR': 'NI', 'TIR': 'TI'}
ROUTE_SOURCES = (
    ('NIR', 'TIR'),
    ('RGB', 'TIR'),
    ('RGB', 'NIR'),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-config', type=Path, required=True)
    parser.add_argument('--baseline-config', type=Path, required=True)
    parser.add_argument('--baseline-checkpoint', type=Path, required=True)
    parser.add_argument('--final-config', type=Path, required=True)
    parser.add_argument('--final-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--batch-size', type=int, default=64)
    return parser.parse_args()


def conditions():
    values = [
        {'name': 'complete', 'kind': 'complete'},
        {'name': 'missing_RGB', 'kind': 'missing', 'modalities': ['RGB']},
        {'name': 'missing_NIR', 'kind': 'missing', 'modalities': ['NIR']},
        {'name': 'missing_TIR', 'kind': 'missing', 'modalities': ['TIR']},
        {
            'name': 'RGB_only', 'kind': 'missing',
            'modalities': ['NIR', 'TIR'],
        },
        {
            'name': 'NIR_only', 'kind': 'missing',
            'modalities': ['RGB', 'TIR'],
        },
        {
            'name': 'TIR_only', 'kind': 'missing',
            'modalities': ['RGB', 'NIR'],
        },
    ]
    for modality in MODALITY_KEYS:
        for ratio in (0.25, 0.50, 0.75):
            values.append({
                'name': 'occlude_{}_{:02d}'.format(
                    modality, int(ratio * 100)),
                'kind': 'occlusion',
                'modality': modality,
                'area_ratio': ratio,
            })
    return values


def center_occlude(tensor, area_ratio):
    result = tensor.clone()
    height, width = result.shape[-2:]
    scale = float(area_ratio) ** 0.5
    block_h = max(1, min(height, round(height * scale)))
    block_w = max(1, min(width, round(width * scale)))
    top = (height - block_h) // 2
    left = (width - block_w) // 2
    # Inputs are normalized; zero is the per-channel dataset mean and avoids
    # introducing an arbitrary out-of-range missing-modality sentinel.
    result[..., top:top + block_h, left:left + block_w] = 0.0
    return result


def transform(images, condition):
    result = dict(images)
    if condition['kind'] == 'missing':
        for modality in condition['modalities']:
            key = MODALITY_KEYS[modality]
            result[key] = torch.zeros_like(images[key])
    elif condition['kind'] == 'occlusion':
        key = MODALITY_KEYS[condition['modality']]
        result[key] = center_occlude(images[key], condition['area_ratio'])
    return result


class Diagnostics:
    def __init__(self, model, condition_names):
        self.model = model
        self.current = None
        self.route_sum = {name: None for name in condition_names}
        self.route_count = {name: 0 for name in condition_names}
        self.mask_selected = {
            name: np.zeros(3, dtype=np.float64) for name in condition_names}
        self.mask_total = {
            name: np.zeros(3, dtype=np.float64) for name in condition_names}
        self.handle = None
        selector = getattr(model, 'HS', None)
        if selector is not None:
            self.handle = selector.register_forward_hook(self._mask_hook)

    def _mask_hook(self, _module, _inputs, output):
        if self.current is None or not isinstance(output, (tuple, list)):
            return
        masks = output[3]
        if not isinstance(masks, (tuple, list)):
            return
        for index, mask in enumerate(masks[:3]):
            self.mask_selected[self.current][index] += float(
                mask.detach().sum().item())
            self.mask_total[self.current][index] += float(mask.numel())

    def collect_routes(self, batch_size):
        aci = getattr(self.model, 'ACI', None)
        if aci is None:
            return
        weights = [
            stage._last_route_weights.detach().float()
            for stage in aci.stages
            if stage._last_route_weights is not None
        ]
        if not weights:
            return
        value = torch.stack(weights, dim=0).sum(dim=1).cpu().numpy()
        if self.route_sum[self.current] is None:
            self.route_sum[self.current] = value
        else:
            self.route_sum[self.current] += value
        self.route_count[self.current] += int(batch_size)

    def summary(self):
        result = {}
        for name in self.route_sum:
            routes = self.route_sum[name]
            route_count = self.route_count[name]
            mask_ratio = np.divide(
                self.mask_selected[name], self.mask_total[name],
                out=np.zeros(3), where=self.mask_total[name] > 0)
            result[name] = {
                'route_weights': (
                    None if routes is None or route_count == 0
                    else (routes / route_count).round(6).tolist()),
                'effective_mask_ratio': {
                    modality: round(float(mask_ratio[index]), 6)
                    for index, modality in enumerate(('RGB', 'NIR', 'TIR'))
                },
            }
        return result

    def close(self):
        if self.handle is not None:
            self.handle.remove()


def evaluate_model(name, args, row_config, checkpoint, condition_specs):
    cfg, model, loader, num_query = build_eval_run(
        args.base_config, row_config, checkpoint,
        seed=args.seed, test_batch_size=args.batch_size)
    if str(cfg.DATASETS.NAMES) != 'RGBNT201':
        raise ValueError('review robustness protocol is fixed to RGBNT201')
    evaluators = {
        item['name']: R1_mAP_eval(
            num_query, max_rank=50, feat_norm=True, reranking=False)
        for item in condition_specs
    }
    for evaluator in evaluators.values():
        evaluator.reset()
    diagnostics = Diagnostics(model, evaluators)
    started = time.monotonic()

    with torch.inference_mode():
        for images, pids, camids, camera_labels, view_labels, paths in loader:
            images = {
                key: value.cuda(non_blocking=True)
                for key, value in images.items()
            }
            camera_labels = camera_labels.cuda(non_blocking=True)
            view_labels = view_labels.cuda(non_blocking=True)
            for condition in condition_specs:
                condition_name = condition['name']
                diagnostics.current = condition_name
                current_images = transform(images, condition)
                feature = model(
                    current_images, cam_label=camera_labels,
                    view_label=view_labels, mode=1, img_path=paths)
                evaluators[condition_name].update(
                    (feature, pids, camids))
                diagnostics.collect_routes(feature.shape[0])

    metric_results = {}
    for condition in condition_specs:
        condition_name = condition['name']
        cmc, mean_ap, *_ = evaluators[condition_name].compute()
        metric_results[condition_name] = {
            'mAP': round(float(mean_ap) * 100.0, 4),
            'Rank1': round(float(cmc[0]) * 100.0, 4),
            'Rank5': round(float(cmc[4]) * 100.0, 4),
            'Rank10': round(float(cmc[9]) * 100.0, 4),
        }
    diagnostic_results = diagnostics.summary()
    diagnostics.close()
    model.cpu()
    del model, loader
    torch.cuda.empty_cache()
    return {
        'name': name,
        'config': str(row_config),
        'checkpoint': str(checkpoint),
        'metrics': metric_results,
        'diagnostics': diagnostic_results,
        'elapsed_seconds': round(time.monotonic() - started, 3),
    }


def main():
    args = parse_args()
    require_cuda()
    for path in (
            args.base_config, args.baseline_config, args.baseline_checkpoint,
            args.final_config, args.final_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    condition_specs = conditions()
    results = [
        evaluate_model(
            'CleanOldT14', args, args.baseline_config,
            args.baseline_checkpoint, condition_specs),
        evaluate_model(
            'Final', args, args.final_config,
            args.final_checkpoint, condition_specs),
    ]
    summary = {
        'dataset': 'RGBNT201',
        'seed': args.seed,
        'test_batch_size': args.batch_size,
        're_ranking': False,
        'missing_encoding': 'zero after input normalization (dataset mean)',
        'occlusion': 'center rectangle set to normalized zero',
        'route_source_order_by_target': {
            target: list(ROUTE_SOURCES[index])
            for index, target in enumerate(('RGB', 'NIR', 'TIR'))
        },
        'conditions': condition_specs,
        'models': results,
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    (args.output_dir / 'DONE').write_text(
        json.dumps({
            'dataset': summary['dataset'],
            'models': [item['name'] for item in results],
        }, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
