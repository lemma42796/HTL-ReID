#!/usr/bin/env python3
"""Generate the inference-only visual analyses used by the HTL-ReID paper."""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.defaults import _C as DEFAULT_CFG
from data import make_dataloader
from modeling import make_model
from utils.metrics import eval_func


MODALITIES = ("RGB", "NIR", "TIR")
ACI_SOURCES = (("NIR", "TIR"), ("RGB", "TIR"), ("RGB", "NIR"))
DHF_ROUTES = ("RGB", "NIR", "TIR", "RGB+NIR", "RGB+TIR", "NIR+TIR", "All")
DHF_ROUTE_MODALITIES = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--full-config", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1111)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def first_attr(obj, *names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError("none of {} exists on {}".format(names, type(obj).__name__))


def build_cfg(base_config, run_config, seed):
    cfg = DEFAULT_CFG.clone()
    cfg.merge_from_file(str(base_config))
    cfg.merge_from_file(str(run_config))
    cfg.SOLVER.SEED = int(seed)
    cfg.TEST.RE_RANKING = "no"
    cfg.TEST.IMS_PER_BATCH = 64
    return cfg


def build_run(base_config, run_config, checkpoint, seed):
    cfg = build_cfg(base_config, run_config, seed)
    loaders = make_dataloader(cfg)
    train_loader, train_loader_normal, val_loader = loaders[:3]
    num_query, num_classes, camera_num = loaders[3:6]
    del train_loader, train_loader_normal
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num).cuda()
    model.load_param(str(checkpoint))
    model.eval()
    return cfg, model, val_loader, num_query


def normalized_cpu(blocks):
    return {name: F.normalize(value.float(), dim=1).cpu() for name, value in blocks.items()}


def canonical_paths(paths, test_dir):
    result = []
    for item in paths:
        if isinstance(item, (tuple, list)):
            result.append(tuple(str(value) for value in item))
        else:
            path = Path(str(item))
            if path.is_absolute():
                result.append((str(path),))
            else:
                result.append(tuple(
                    str(test_dir / modality / path.name)
                    for modality in ("RGB", "NI", "TI")))
    return result


def install_full_hooks(model):
    capture = {
        "part_masks": [],
        "shared_mask": None,
        "dhf_features": None,
    }
    handles = []
    selector = first_attr(model, "SFTS", "HS")
    dhf = first_attr(model, "DECOUPLED_MOE", "DHF")

    def part_hook(_module, _inputs, output):
        if len(capture["part_masks"]) >= 3:
            return
        mask = output[0] if isinstance(output, (tuple, list)) else output
        capture["part_masks"].append(mask[:1].detach().cpu())

    def selector_hook(_module, _inputs, output):
        if capture["shared_mask"] is not None:
            return
        if not isinstance(output, (tuple, list)) or len(output) < 4:
            return
        masks = output[3]
        if isinstance(masks, (tuple, list)) and masks:
            capture["shared_mask"] = masks[0][:1].detach().cpu()

    def dhf_pre_hook(_module, inputs):
        if capture["dhf_features"] is None:
            capture["dhf_features"] = tuple(value[:1].detach().cpu() for value in inputs[:3])

    handles.append(selector.part_select.register_forward_hook(part_hook))
    handles.append(selector.register_forward_hook(selector_hook))
    handles.append(dhf.register_forward_pre_hook(dhf_pre_hook))
    return capture, handles


def full_features(model, loader, test_dir):
    original = {name: [] for name in ("aci", "original", "moe")}
    flipped = {name: [] for name in ("aci", "original", "moe")}
    pids, camids, all_paths = [], [], []
    aci_values, dhf_values = [], []
    capture, handles = install_full_hooks(model)
    aci = first_attr(model, "ACI", "FACR")
    dhf = first_attr(model, "DHF", "DECOUPLED_MOE")

    with torch.inference_mode():
        for images, vid, camid, camera_labels, view_labels, paths in loader:
            images = {name: tensor.cuda(non_blocking=True) for name, tensor in images.items()}
            camera_labels = camera_labels.cuda(non_blocking=True)
            view_labels = view_labels.cuda(non_blocking=True)
            blocks = model(
                images, cam_label=camera_labels, view_label=view_labels,
                mode=1, img_path=paths, return_descriptor_components=True)
            key = "aci" if "aci" in blocks else "facr"
            selected = normalized_cpu({
                "aci": blocks[key],
                "original": blocks["original"],
                "moe": blocks["moe"],
            })
            for name in original:
                original[name].append(selected[name])

            stage_weights = [stage._last_route_weights.detach().float().cpu() for stage in aci.stages]
            aci_values.append(torch.stack(stage_weights, dim=1))
            dhf_values.append(dhf._last_gate.detach().float().cpu())

            flipped_images = {name: torch.flip(tensor, dims=(3,)) for name, tensor in images.items()}
            flip_blocks = model(
                flipped_images, cam_label=camera_labels, view_label=view_labels,
                mode=1, img_path=paths, return_descriptor_components=True)
            key = "aci" if "aci" in flip_blocks else "facr"
            selected = normalized_cpu({
                "aci": flip_blocks[key],
                "original": flip_blocks["original"],
                "moe": flip_blocks["moe"],
            })
            for name in flipped:
                flipped[name].append(selected[name])

            pids.extend(np.asarray(vid).tolist())
            camids.extend(np.asarray(camid).tolist())
            all_paths.extend(canonical_paths(paths, test_dir))

    for handle in handles:
        handle.remove()
    original = {name: torch.cat(values) for name, values in original.items()}
    flipped = {name: torch.cat(values) for name, values in flipped.items()}
    blended = {
        name: F.normalize(0.1 * original[name] + 0.9 * flipped[name], dim=1)
        for name in original
    }
    feature = F.normalize(torch.cat(
        (blended["aci"], 1.1 * blended["original"], 1.58 * blended["moe"]), dim=1), dim=1)
    return {
        "feature": feature,
        "pids": np.asarray(pids),
        "camids": np.asarray(camids),
        "paths": all_paths,
        "aci": torch.cat(aci_values).numpy(),
        "dhf": torch.cat(dhf_values).numpy(),
        "capture": capture,
    }


def baseline_features(model, loader, test_dir):
    features, pids, camids, all_paths = [], [], [], []
    with torch.inference_mode():
        for images, vid, camid, camera_labels, view_labels, paths in loader:
            images = {name: tensor.cuda(non_blocking=True) for name, tensor in images.items()}
            camera_labels = camera_labels.cuda(non_blocking=True)
            view_labels = view_labels.cuda(non_blocking=True)
            feature = model(
                images, cam_label=camera_labels, view_label=view_labels,
                mode=1, img_path=paths)
            features.append(F.normalize(feature.float(), dim=1).cpu())
            pids.extend(np.asarray(vid).tolist())
            camids.extend(np.asarray(camid).tolist())
            all_paths.extend(canonical_paths(paths, test_dir))
    return {
        "feature": torch.cat(features),
        "pids": np.asarray(pids),
        "camids": np.asarray(camids),
        "paths": all_paths,
    }


def distance_matrix(feature, num_query):
    query = feature[:num_query].cuda(non_blocking=True)
    gallery = feature[num_query:].cuda(non_blocking=True)
    distance = (2.0 - 2.0 * query @ gallery.t()).clamp_min_(0.0)
    result = distance.cpu().numpy()
    del query, gallery, distance
    return result


def metrics(distance, pids, camids, num_query):
    cmc, mean_ap = eval_func(
        distance, pids[:num_query], pids[num_query:],
        camids[:num_query], camids[num_query:], max_rank=50)
    return {
        "mAP": round(float(mean_ap) * 100.0, 4),
        "Rank1": round(float(cmc[0]) * 100.0, 4),
        "Rank5": round(float(cmc[4]) * 100.0, 4),
        "Rank10": round(float(cmc[9]) * 100.0, 4),
    }


def read_image(path):
    return Image.open(path).convert("RGB")


def grid_shape(tokens, width, height):
    choices = []
    for rows in range(1, int(math.sqrt(tokens)) + 1):
        if tokens % rows == 0:
            cols = tokens // rows
            for r, c in ((rows, cols), (cols, rows)):
                choices.append((abs((c / r) - (width / height)), r, c))
    return min(choices)[1:]


def mask_heat(mask, image, cmap="jet"):
    values = np.asarray(mask, dtype=np.float32).reshape(-1)
    rows, cols = grid_shape(values.size, image.width, image.height)
    values = values.reshape(rows, cols)
    low, high = float(values.min()), float(values.max())
    if high > low:
        values = (values - low) / (high - low)
    rgba = plt.get_cmap(cmap)(values)
    overlay = Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8)).resize(
        image.size, Image.Resampling.NEAREST)
    return Image.blend(image, overlay, 0.48)


def save_hs_figure(capture, paths, output):
    part_masks = capture["part_masks"]
    shared = capture["shared_mask"]
    if len(part_masks) != 3 or shared is None:
        raise RuntimeError("HS hooks did not capture the expected masks")
    modality_masks = [value[0].numpy().astype(bool) for value in part_masks]
    saliency_union = np.logical_or.reduce(modality_masks)
    final_mask = shared[0].numpy().astype(bool)
    frequency_added = final_mask & ~saliency_union
    images = [read_image(path) for path in paths]
    fig, axes = plt.subplots(3, 4, figsize=(11.5, 10.5))
    titles = ("Input", "Saliency selection", "Frequency additions", "Final HS mask")
    for row, (name, image, mask) in enumerate(zip(MODALITIES, images, modality_masks)):
        panels = (
            image,
            mask_heat(mask.astype(float), image),
            mask_heat(frequency_added.astype(float), image),
            mask_heat(final_mask.astype(float), image),
        )
        for col, panel in enumerate(panels):
            axes[row, col].imshow(panel)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=11)
            if col == 0:
                axes[row, col].set_ylabel(name, fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_aci_figure(values, output):
    mean = values.mean(axis=0)  # stages, targets, two sources
    stages = mean.shape[0]
    fig, axes = plt.subplots(1, stages, figsize=(4.0 * stages, 3.8), squeeze=False)
    for stage in range(stages):
        matrix = np.full((3, 3), np.nan, dtype=np.float32)
        for target in range(3):
            for index, source_name in enumerate(ACI_SOURCES[target]):
                source = MODALITIES.index(source_name)
                matrix[target, source] = mean[stage, target, index]
        shown = np.nan_to_num(matrix, nan=0.0)
        ax = axes[0, stage]
        image = ax.imshow(shown, vmin=0.0, vmax=1.0, cmap="viridis")
        for i in range(3):
            for j in range(3):
                text = "—" if i == j else "{:.2f}".format(matrix[i, j])
                ax.text(j, i, text, ha="center", va="center", color="white", fontsize=10)
        ax.set_xticks(range(3), MODALITIES)
        ax.set_yticks(range(3), MODALITIES)
        ax.set_xlabel("Source modality")
        if stage == 0:
            ax.set_ylabel("Target modality")
        ax.set_title("ACI stage {}".format(stage + 1))
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03, label="Mean routing weight")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_dhf_gate_figure(values, output):
    per_sample = values.mean(axis=1)
    order = np.argsort(per_sample.argmax(axis=1), kind="stable")
    shown = per_sample[order[:min(80, len(order))]]
    average = per_sample.mean(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": (1.7, 1)})
    image = axes[0].imshow(shown, aspect="auto", vmin=0.0, vmax=max(0.25, shown.max()), cmap="magma")
    axes[0].set_xticks(range(7), DHF_ROUTES, rotation=35, ha="right")
    axes[0].set_ylabel("Test samples")
    axes[0].set_title("Sample-adaptive DHF routing")
    fig.colorbar(image, ax=axes[0], fraction=0.035, pad=0.03)
    axes[1].bar(range(7), average, color="#4472C4")
    axes[1].set_xticks(range(7), DHF_ROUTES, rotation=35, ha="right")
    axes[1].set_ylabel("Mean gate weight")
    axes[1].set_title("Dataset mean")
    axes[1].set_ylim(0, max(0.2, float(average.max()) * 1.2))
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def dhf_attention_maps(model, capture):
    dhf = first_attr(model, "DHF", "DECOUPLED_MOE")
    features = tuple(value.cuda() for value in capture["dhf_features"])
    contexts, _anchors = dhf._route_inputs(*features)
    maps = []
    with torch.inference_mode():
        for index, context in enumerate(contexts):
            query = dhf.route_tokens[index:index + 1]
            _output, weight = dhf.route_attn[index](
                query, context, context, need_weights=True,
                average_attn_weights=False)
            maps.append(weight.mean(dim=1)[0, 0].float().cpu().numpy())
    return maps, [value.shape[1] for value in features]


def save_dhf_activation_figure(model, capture, paths, output):
    maps, lengths = dhf_attention_maps(model, capture)
    images = [read_image(path) for path in paths]
    fig, axes = plt.subplots(7, 3, figsize=(8.5, 18))
    for route, modalities in enumerate(DHF_ROUTE_MODALITIES):
        cursor = 0
        route_maps = {}
        for modality in modalities:
            length = lengths[modality]
            route_maps[modality] = maps[route][cursor + 1:cursor + length]
            cursor += length
        for modality in range(3):
            ax = axes[route, modality]
            if modality in route_maps:
                ax.imshow(mask_heat(route_maps[modality], images[modality]))
            else:
                ax.imshow(images[modality].convert("L"), cmap="gray", alpha=0.25)
            ax.axis("off")
            if route == 0:
                ax.set_title(MODALITIES[modality])
            if modality == 0:
                ax.text(
                    0.03, 0.04, DHF_ROUTES[route], transform=ax.transAxes,
                    color="white", fontsize=9, weight="bold",
                    bbox={"facecolor": "black", "alpha": 0.72, "pad": 2},
                    ha="left", va="bottom")
    fig.suptitle("DHF route activation maps", y=0.995, fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def pairwise_squared(x):
    return np.maximum(0.0, np.sum(x * x, axis=1, keepdims=True) +
                      np.sum(x * x, axis=1)[None, :] - 2.0 * x @ x.T)


def tsne_probabilities(x, perplexity=15.0):
    distances = pairwise_squared(x)
    n = len(x)
    conditional = np.zeros((n, n), dtype=np.float64)
    target = math.log(perplexity)
    for i in range(n):
        mask = np.arange(n) != i
        row = distances[i, mask]
        beta, low, high = 1.0, -np.inf, np.inf
        for _ in range(50):
            values = np.exp(-row * beta)
            total = max(values.sum(), 1e-12)
            entropy = math.log(total) + beta * float((row * values).sum()) / total
            difference = entropy - target
            if abs(difference) < 1e-5:
                break
            if difference > 0:
                low = beta
                beta = beta * 2.0 if np.isinf(high) else (beta + high) / 2.0
            else:
                high = beta
                beta = beta / 2.0 if np.isinf(low) else (beta + low) / 2.0
        conditional[i, mask] = values / total
    probabilities = conditional + conditional.T
    probabilities /= max(probabilities.sum(), 1e-12)
    return np.maximum(probabilities, 1e-12)


def tsne(x, seed):
    x = np.asarray(x, dtype=np.float64)
    x -= x.mean(axis=0, keepdims=True)
    gram = x @ x.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:min(30, len(x) - 1)]
    x = eigenvectors[:, order] * np.sqrt(np.maximum(eigenvalues[order], 0.0))[None, :]
    probabilities = tsne_probabilities(x)
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, 1e-4, size=(len(x), 2))
    velocity = np.zeros_like(y)
    gains = np.ones_like(y)
    for iteration in range(700):
        numerator = 1.0 / (1.0 + pairwise_squared(y))
        np.fill_diagonal(numerator, 0.0)
        q = np.maximum(numerator / max(numerator.sum(), 1e-12), 1e-12)
        p = probabilities * (4.0 if iteration < 100 else 1.0)
        affinity = (p - q) * numerator
        gradient = 4.0 * np.sum(
            affinity[:, :, None] * (y[:, None, :] - y[None, :, :]), axis=1)
        changed = np.sign(gradient) != np.sign(velocity)
        gains = np.where(changed, gains + 0.2, gains * 0.8)
        gains = np.maximum(gains, 0.01)
        momentum = 0.5 if iteration < 250 else 0.8
        velocity = momentum * velocity - 200.0 * gains * gradient
        y += velocity
        y -= y.mean(axis=0, keepdims=True)
    return y


def select_tsne_indices(pids, paths, num_query, max_ids=10, per_id=7):
    gallery = np.arange(num_query, len(pids))
    selected = []
    for pid in sorted(set(pids[gallery].tolist())):
        candidates = [index for index in gallery if pids[index] == pid]
        unique, seen = [], set()
        for index in candidates:
            key = paths[index][0]
            if key not in seen:
                unique.append(index)
                seen.add(key)
        if len(unique) >= 4:
            selected.extend(unique[:per_id])
        if len(set(pids[selected].tolist())) >= max_ids:
            break
    return np.asarray(selected, dtype=np.int64)


def save_tsne_figure(baseline, full, pids, paths, num_query, seed, output):
    indices = select_tsne_indices(pids, paths, num_query)
    labels = pids[indices]
    base_xy = tsne(baseline[indices].numpy(), seed)
    full_xy = tsne(full[indices].numpy(), seed)
    identities = sorted(set(labels.tolist()))
    colors = plt.get_cmap("tab10", len(identities))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for ax, points, title in zip(axes, (base_xy, full_xy), ("Backbone", "Full HTL-ReID")):
        for color_index, pid in enumerate(identities):
            mask = labels == pid
            ax.scatter(points[mask, 0], points[mask, 1], s=30,
                       color=colors(color_index), label=str(pid), alpha=0.85)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].legend(title="Vehicle ID", bbox_to_anchor=(1.03, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def valid_ranking(distance_row, q_index, pids, camids, num_query):
    indices = np.argsort(distance_row)
    q_pid, q_camid = pids[q_index], camids[q_index]
    keep = ~((pids[num_query:][indices] == q_pid) &
             (camids[num_query:][indices] == q_camid))
    return indices[keep]


def first_match_rank(order, q_pid, gallery_pids):
    matches = np.flatnonzero(gallery_pids[order] == q_pid)
    return int(matches[0]) if len(matches) else 10 ** 9


def thumbnail(path, correct, size=(88, 176)):
    image = read_image(path)
    image.thumbnail((size[0] - 6, size[1] - 26), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    left = (size[0] - image.width) // 2
    canvas.paste(image, (left, 20))
    draw = ImageDraw.Draw(canvas)
    color = "#2ca02c" if correct else "#d62728"
    draw.rectangle((1, 1, size[0] - 2, size[1] - 2), outline=color, width=4)
    return canvas


def save_ranklist_figure(base_distance, full_distance, pids, camids, paths, num_query, output):
    gallery_pids = pids[num_query:]
    candidates = []
    for q_index in range(num_query):
        base_order = valid_ranking(base_distance[q_index], q_index, pids, camids, num_query)
        full_order = valid_ranking(full_distance[q_index], q_index, pids, camids, num_query)
        base_rank = first_match_rank(base_order, pids[q_index], gallery_pids)
        full_rank = first_match_rank(full_order, pids[q_index], gallery_pids)
        if full_rank < base_rank and full_rank < 5:
            candidates.append((base_rank - full_rank, q_index, base_order, full_order))
    if not candidates:
        for q_index in range(num_query):
            base_order = valid_ranking(base_distance[q_index], q_index, pids, camids, num_query)
            full_order = valid_ranking(full_distance[q_index], q_index, pids, camids, num_query)
            candidates.append((
                first_match_rank(base_order, pids[q_index], gallery_pids) -
                first_match_rank(full_order, pids[q_index], gallery_pids),
                q_index, base_order, full_order))
    candidates.sort(reverse=True, key=lambda item: item[0])
    chosen, chosen_pids = [], set()
    for candidate in candidates:
        q_pid = int(pids[candidate[1]])
        if q_pid in chosen_pids:
            continue
        chosen.append(candidate)
        chosen_pids.add(q_pid)
        if len(chosen) == 4:
            break
    cell_w, cell_h = 88, 176
    label_w = 96
    canvas = Image.new("RGB", (label_w + 11 * cell_w, len(chosen) * 2 * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row, (_gain, q_index, base_order, full_order) in enumerate(chosen):
        for variant, order in enumerate((base_order, full_order)):
            y = (row * 2 + variant) * cell_h
            label = "Backbone" if variant == 0 else "Full"
            draw.text((7, y + 62), label, fill="black", font=font)
            if variant == 0:
                query = thumbnail(paths[q_index][0], True, (cell_w, cell_h))
                ImageDraw.Draw(query).text((7, 5), "Query", fill="black", font=font)
                canvas.paste(query, (label_w, y))
            else:
                canvas.paste(Image.new("RGB", (cell_w, cell_h), "#f4f4f4"), (label_w, y))
            for rank, gallery_local in enumerate(order[:10]):
                gallery_index = num_query + int(gallery_local)
                correct = pids[gallery_index] == pids[q_index]
                cell = thumbnail(paths[gallery_index][0], correct, (cell_w, cell_h))
                ImageDraw.Draw(cell).text((7, 5), str(rank + 1), fill="black", font=font)
                canvas.paste(cell, (label_w + (rank + 1) * cell_w, y))
    canvas.save(output)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    for path in (
        args.base_config, args.baseline_config, args.full_config,
        args.baseline_checkpoint, args.full_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    started = time.monotonic()

    print("[1/6] extracting full-model descriptors and internal weights", flush=True)
    full_cfg, full_model, full_loader, num_query = build_run(
        args.base_config, args.full_config, args.full_checkpoint, args.seed)
    dataset_name = full_cfg.DATASETS.NAMES
    if isinstance(dataset_name, (tuple, list)):
        dataset_name = dataset_name[0]
    test_dir = Path(full_cfg.DATASETS.ROOT_DIR) / str(dataset_name) / "test"
    full = full_features(full_model, full_loader, test_dir)
    full_distance = distance_matrix(full["feature"], num_query)
    full_metric = metrics(full_distance, full["pids"], full["camids"], num_query)

    print("[2/6] writing HS, ACI, and DHF figures", flush=True)
    first_paths = full["paths"][0]
    save_hs_figure(full["capture"], first_paths, args.output_dir / "fig_hs_selection.png")
    save_aci_figure(full["aci"], args.output_dir / "fig_aci_routing.png")
    save_dhf_gate_figure(full["dhf"], args.output_dir / "fig_dhf_gating.png")
    save_dhf_activation_figure(
        full_model, full["capture"], first_paths,
        args.output_dir / "fig_dhf_activation.png")
    del full_loader
    full_model.cpu()
    del full_model
    torch.cuda.empty_cache()

    print("[3/6] extracting backbone descriptors", flush=True)
    baseline_cfg, baseline_model, baseline_loader, baseline_num_query = build_run(
        args.base_config, args.baseline_config, args.baseline_checkpoint, args.seed)
    if baseline_num_query != num_query:
        raise RuntimeError("baseline and full query counts differ")
    baseline = baseline_features(baseline_model, baseline_loader, test_dir)
    if not (np.array_equal(baseline["pids"], full["pids"]) and
            np.array_equal(baseline["camids"], full["camids"])):
        raise RuntimeError("baseline and full evaluation orders differ")
    baseline_distance = distance_matrix(baseline["feature"], num_query)
    baseline_metric = metrics(
        baseline_distance, baseline["pids"], baseline["camids"], num_query)
    baseline_model.cpu()
    del baseline_model, baseline_loader
    torch.cuda.empty_cache()

    print("[4/6] writing t-SNE", flush=True)
    save_tsne_figure(
        baseline["feature"], full["feature"], full["pids"], full["paths"],
        num_query, args.seed, args.output_dir / "fig_tsne_baseline_full.png")
    print("[5/6] writing rank list", flush=True)
    save_ranklist_figure(
        baseline_distance, full_distance, full["pids"], full["camids"],
        full["paths"], num_query,
        args.output_dir / "fig_ranklist_baseline_full.png")

    summary = {
        "dataset": "RGBNT201",
        "seed": args.seed,
        "test_batch_size": 64,
        "re_ranking": False,
        "num_query": int(num_query),
        "num_total": int(len(full["pids"])),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "full_checkpoint": str(args.full_checkpoint),
        "full_inference": {
            "flip_alpha": 0.9,
            "weights": {"ACI": 1.0, "CLS": 1.1, "PLR": 0.0, "DHF": 1.58},
        },
        "metrics": {"baseline": baseline_metric, "full": full_metric},
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.output_dir / "routing_statistics.npz",
        aci=full["aci"], dhf=full["dhf"], pids=full["pids"], camids=full["camids"])
    print("[6/6] done", flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
