#!/usr/bin/env python3
"""Generate clean visual evidence for the frozen final model."""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from generate_paper_visualizations import (
    canonical_paths,
    mask_heat,
    save_aci_figure,
)
from review_evidence_common import build_eval_run, require_cuda, set_seed


MODALITIES = ('RGB', 'NIR', 'TIR')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-config', type=Path, required=True)
    parser.add_argument('--final-config', type=Path, required=True)
    parser.add_argument('--final-checkpoint', type=Path, required=True)
    parser.add_argument('--robustness-summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--batch-size', type=int, default=64)
    return parser.parse_args()


def install_hs_hooks(model):
    capture = {'raw_masks': [], 'effective_masks': None}
    handles = []
    selector = getattr(model, 'HS', None)
    if selector is None:
        raise RuntimeError('final visualization requires HS')

    def part_hook(_module, _inputs, output):
        if len(capture['raw_masks']) >= 3:
            return
        mask = output[0] if isinstance(output, (tuple, list)) else output
        capture['raw_masks'].append(mask[:1].detach().cpu())

    def selector_hook(_module, _inputs, output):
        if capture['effective_masks'] is not None:
            return
        masks = output[3]
        capture['effective_masks'] = tuple(
            value[:1].detach().cpu() for value in masks)

    handles.append(selector.part_select.register_forward_hook(part_hook))
    handles.append(selector.register_forward_hook(selector_hook))
    return capture, handles


def extract(model, loader, test_dir, capture_internal=False):
    pids, camids, paths = [], [], []
    route_values = []
    capture, handles = ({}, [])
    if capture_internal:
        capture, handles = install_hs_hooks(model)
    with torch.inference_mode():
        for images, vid, camid, camera_labels, view_labels, batch_paths in loader:
            images = {
                key: value.cuda(non_blocking=True)
                for key, value in images.items()
            }
            camera_labels = camera_labels.cuda(non_blocking=True)
            view_labels = view_labels.cuda(non_blocking=True)
            feature = model(
                images, cam_label=camera_labels, view_label=view_labels,
                mode=1, img_path=batch_paths)
            pids.extend(np.asarray(vid).tolist())
            camids.extend(np.asarray(camid).tolist())
            paths.extend(canonical_paths(batch_paths, test_dir))
            if capture_internal:
                weights = [
                    stage._last_route_weights.detach().float().cpu()
                    for stage in model.ACI.stages
                ]
                route_values.append(
                    torch.stack(weights, dim=1))
    for handle in handles:
        handle.remove()
    return {
        'pids': np.asarray(pids),
        'camids': np.asarray(camids),
        'paths': paths,
        'routes': (
            None if not route_values
            else torch.cat(route_values).numpy()),
        'capture': capture,
    }


def save_hs_figure(capture, paths, consensus_specific, output):
    raw_values = capture.get('raw_masks', [])
    effective_values = capture.get('effective_masks')
    if len(raw_values) != 3 or effective_values is None:
        raise RuntimeError('HS hooks did not capture three modality masks')
    raw = [value[0].numpy().astype(bool) for value in raw_values]
    effective = [value[0].numpy().astype(bool) for value in effective_values]
    votes = np.stack(raw).sum(axis=0)
    consensus = votes >= 2
    union = votes >= 1
    if consensus_specific:
        broadcast = consensus
        specifics = [value & ~consensus for value in raw]
    else:
        broadcast = union
        specifics = [np.zeros_like(union) for _ in raw]
    structural = [broadcast | value for value in specifics]
    frequency = [
        effective[index] & ~structural[index] for index in range(3)]
    images = [Image.open(path).convert('RGB') for path in paths]
    fig, axes = plt.subplots(
        3, 6, figsize=(11.8, 9.0),
        gridspec_kw={'wspace': 0.08, 'hspace': 0.06})
    broadcast_title = (
        'Consensus broadcast' if consensus_specific else 'Union broadcast')
    titles = (
        'Input', 'Per-modality selection', broadcast_title,
        'Modality-specific', 'Frequency additions', 'Effective mask')
    for row, (name, image) in enumerate(zip(MODALITIES, images)):
        panels = (
            image,
            mask_heat(raw[row].astype(float), image),
            mask_heat(broadcast.astype(float), image),
            mask_heat(specifics[row].astype(float), image),
            mask_heat(frequency[row].astype(float), image),
            mask_heat(effective[row].astype(float), image),
        )
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel)
            axes[row, column].axis('off')
            if row == 0:
                axes[row, column].set_title(titles[column], fontsize=10)
            if column == 0:
                axes[row, column].set_ylabel(name, fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01)
    fig.savefig(output, dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_robustness_figure(summary, output):
    final = next(
        item for item in summary['models'] if item['name'] == 'Final')
    baseline = next(
        item for item in summary['models']
        if item['name'] == 'CleanOldT14')
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    missing = ('missing_RGB', 'missing_NIR', 'missing_TIR')
    x = np.arange(len(missing))
    width = 0.36
    axes[0].bar(
        x - width / 2,
        [baseline['metrics'][name]['mAP'] for name in missing],
        width, label='Reference', color='#9E9E9E')
    axes[0].bar(
        x + width / 2,
        [final['metrics'][name]['mAP'] for name in missing],
        width, label='Final', color='#4472C4')
    axes[0].set_xticks(x, ('No RGB', 'No NIR', 'No TIR'))
    axes[0].set_ylabel('mAP (%)')
    axes[0].set_title('Missing-modality robustness')
    axes[0].legend()

    ratios = (0, 25, 50, 75)
    for modality, color in zip(
            MODALITIES, ('#D62728', '#2CA02C', '#1F77B4')):
        values = [final['metrics']['complete']['mAP']] + [
            final['metrics']['occlude_{}_{:02d}'.format(
                modality, ratio)]['mAP'] for ratio in ratios[1:]
        ]
        axes[1].plot(
            ratios, values, marker='o', label=modality, color=color)
    axes[1].set_xlabel('Centered occlusion area (%)')
    axes[1].set_ylabel('mAP (%)')
    axes[1].set_title('Controlled single-modality degradation')
    axes[1].set_xticks(ratios)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    require_cuda()
    for path in (
            args.base_config, args.final_config, args.final_checkpoint,
            args.robustness_summary):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    started = time.monotonic()

    final_cfg, final_model, final_loader, num_query = build_eval_run(
        args.base_config, args.final_config, args.final_checkpoint,
        seed=args.seed, test_batch_size=args.batch_size)
    dataset_name = str(final_cfg.DATASETS.NAMES)
    test_dir = Path(final_cfg.DATASETS.ROOT_DIR) / dataset_name / 'test'
    final = extract(
        final_model, final_loader, test_dir, capture_internal=True)
    save_hs_figure(
        final['capture'], final['paths'][0],
        bool(final_cfg.MODEL.HS_CONSENSUS_SPECIFIC),
        args.output_dir / 'fig_hs_consensus_specific.png')
    save_aci_figure(
        final['routes'], args.output_dir / 'fig_facr_routing.png')
    final_model.cpu()
    del final_model, final_loader
    torch.cuda.empty_cache()
    robustness = json.loads(
        args.robustness_summary.read_text(encoding='utf-8'))
    save_robustness_figure(
        robustness, args.output_dir / 'fig_robustness.png')

    summary = {
        'dataset': 'RGBNT201',
        'seed': args.seed,
        'test_batch_size': args.batch_size,
        're_ranking': False,
        'tta': False,
        'final_config': str(args.final_config),
        'final_checkpoint': str(args.final_checkpoint),
        'num_query': int(num_query),
        'num_total': int(len(final['pids'])),
        'robustness_summary': str(args.robustness_summary),
        'elapsed_seconds': round(time.monotonic() - started, 3),
        'figures': sorted(path.name for path in args.output_dir.glob('*.png')),
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    np.savez_compressed(
        args.output_dir / 'routing_statistics.npz',
        facr=final['routes'], pids=final['pids'], camids=final['camids'])
    (args.output_dir / 'DONE').write_text(
        json.dumps(summary, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
