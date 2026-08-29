#!/usr/bin/env python3
"""Measure reproducible inference efficiency for baseline and final models."""

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from review_evidence_common import build_cfg, require_cuda, set_seed
from modeling import make_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-config', type=Path, required=True)
    parser.add_argument('--baseline-config', type=Path, required=True)
    parser.add_argument('--final-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=50)
    return parser.parse_args()


def profiler_flops(model, inputs, camera, view):
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
            activities=activities, record_shapes=True,
            profile_memory=False, with_flops=True) as profile:
        with torch.inference_mode():
            model(inputs, cam_label=camera, view_label=view)
    return int(sum(int(item.flops or 0) for item in profile.key_averages()))


def measure(name, base_config, row_config, args):
    cfg = build_cfg(base_config, row_config, seed=args.seed)
    cfg.MODEL.PRETRAIN_CHOICE = 'self'
    cfg.MODEL.PRETRAIN_PATH_T = ''
    model = make_model(cfg, num_class=171, camera_num=4).cuda().eval()
    height, width = (int(value) for value in cfg.INPUT.SIZE_TEST)
    inputs = {
        modality: torch.randn(
            args.batch_size, 3, height, width, device='cuda')
        for modality in ('RGB', 'NI', 'TI')
    }
    camera = torch.zeros(args.batch_size, dtype=torch.long, device='cuda')
    view = torch.zeros(args.batch_size, dtype=torch.long, device='cuda')

    with torch.inference_mode():
        descriptor = model(inputs, cam_label=camera, view_label=view)
    flops = profiler_flops(
        model,
        {key: value[:1] for key, value in inputs.items()},
        camera[:1], view[:1])

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(inputs, cam_label=camera, view_label=view)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.repeats):
            model(inputs, cam_label=camera, view_label=view)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    samples = args.batch_size * args.repeats
    result = {
        'name': name,
        'config': str(row_config),
        'parameters': int(sum(value.numel() for value in model.parameters())),
        'trainable_parameters': int(sum(
            value.numel() for value in model.parameters()
            if value.requires_grad)),
        'profiler_flops_per_triplet': flops,
        'profiler_gflops_per_triplet': round(flops / 1e9, 4),
        'descriptor_dimension': int(descriptor.shape[1]),
        'benchmark_batch_size': args.batch_size,
        'warmup_iterations': args.warmup,
        'timed_iterations': args.repeats,
        'elapsed_seconds': round(elapsed, 6),
        'fps_triplets': round(samples / elapsed, 4),
        'latency_ms_per_triplet': round(1000.0 * elapsed / samples, 4),
        'peak_allocated_mib': round(
            torch.cuda.max_memory_allocated() / 2 ** 20, 3),
        'peak_reserved_mib': round(
            torch.cuda.max_memory_reserved() / 2 ** 20, 3),
        'input_size': [height, width],
    }
    model.cpu()
    del model, inputs, descriptor
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    require_cuda()
    for path in (args.base_config, args.baseline_config, args.final_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    results = [
        measure('Backbone', args.base_config, args.baseline_config, args),
        measure('Final', args.base_config, args.final_config, args),
    ]
    summary = {
        'seed': args.seed,
        're_ranking': False,
        'device': torch.cuda.get_device_name(0),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'python_version': platform.python_version(),
        'results': results,
    }
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    (args.output_dir / 'DONE').write_text(
        json.dumps(summary, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
