#!/usr/bin/env python3
"""Plan-C determinism attribution chain: E039, then E040/E041.

Step 1 (E039) reruns the E038 batch-64 train25 protocol with seed 1111 but
SOLVER.STRICT_DETERMINISM=0, isolating the deterministic-algorithm cost from
the seeded sampling trajectory. Steps 2-3 run seeds 2222 and 3333 with the
winning mode: relaxed only if E039 improves over the strict E038 mAP (62.05)
by at least RELAX_GAIN_THRESHOLD; otherwise strict remains the baseline mode.

Each step is executed through tools/run_rgbnt201_fusion.py, which enforces its
own `timeout --signal=TERM --kill-after=10s 30m` per training process.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "tools" / "run_rgbnt201_fusion.py"
OUTPUT_ROOT = Path("/root/autodl-tmp/outputs/HTL-ReID")
PYTHON = "/root/miniconda3/bin/python"

STRICT_CONFIG = "configs/RGBNT201/fusion/t12_sfts_k1_shared_token_recon_b64_25e.yml"
RELAXED_CONFIG = "configs/RGBNT201/fusion/t12_sfts_k1_shared_token_recon_b64_25e_relaxed.yml"
STRICT_BASELINE_MAP = 62.05  # E038 result, strict mode seed 1111
RELAX_GAIN_THRESHOLD = 2.0

STEPS = (
    {
        "experiment": "E039",
        "row": "T12-B64-RELAX25",
        "config": RELAXED_CONFIG,
        "seed": 1111,
        "output": "E039_T12_b64_train25_relaxed_seed1111",
    },
    {
        "experiment": "E040",
        "row": "T12-B64-CHAIN25",
        "config": None,  # decided after E039
        "seed": 2222,
        "output": None,
    },
    {
        "experiment": "E041",
        "row": "T12-B64-CHAIN25",
        "config": None,  # decided after E039
        "seed": 3333,
        "output": None,
    },
)


def run_step(step):
    output_dir = OUTPUT_ROOT / step["output"]
    if output_dir.joinpath("DONE").exists():
        print("[chain] {} already DONE, skipping".format(step["experiment"]))
        return json.loads(
            output_dir.joinpath("run_result.json").read_text(encoding="utf-8"))
    command = [
        PYTHON, str(RUNNER),
        "--single-experiment", step["experiment"],
        "--single-row", step["row"],
        "--single-config", step["config"],
        "--single-output-name", step["output"],
        "--seed", str(step["seed"]),
        "--expected-train-epochs", "25",
        "--expected-base-lr", "0.00035",
        "--expected-batch-size", "64",
    ]
    print("[chain] {} start: {}".format(step["experiment"], " ".join(command)))
    completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    result_path = output_dir / "run_result.json"
    if not result_path.exists():
        raise RuntimeError(
            "{} finished with returncode {} but has no run_result.json".format(
                step["experiment"], completed.returncode))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    print("[chain] {} -> {} mAP={} Rank1={}".format(
        step["experiment"], result["status"], result["mAP"], result["Rank1"]))
    if result["status"] != "completed":
        raise RuntimeError(
            "{} did not complete: {}".format(step["experiment"], result["status"]))
    return result


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    chain_log = []

    e039 = run_step(STEPS[0])
    chain_log.append({"experiment": "E039", "result": e039})
    e039_map = float(e039["mAP"])
    use_relaxed = e039_map >= STRICT_BASELINE_MAP + RELAX_GAIN_THRESHOLD
    mode = "relaxed" if use_relaxed else "strict"
    config = RELAXED_CONFIG if use_relaxed else STRICT_CONFIG
    print("[chain] E039 mAP {:.2f} vs strict baseline {:.2f} (+{:.2f}); "
          "threshold +{:.2f}; selecting '{}' mode for E040/E041".format(
              e039_map, STRICT_BASELINE_MAP, e039_map - STRICT_BASELINE_MAP,
              RELAX_GAIN_THRESHOLD, mode))
    chain_log.append({
        "decision": {
            "mode": mode,
            "e039_mAP": e039_map,
            "strict_baseline_mAP": STRICT_BASELINE_MAP,
            "relax_gain_threshold": RELAX_GAIN_THRESHOLD,
        },
    })

    for step in STEPS[1:]:
        step = dict(step)
        step["config"] = config
        step["output"] = "E{}_T12_b64_train25_{}_seed{}".format(
            step["experiment"][1:], mode, step["seed"])
        result = run_step(step)
        chain_log.append({"experiment": step["experiment"], "result": result})

    OUTPUT_ROOT.joinpath("E039-E041_chain.json").write_text(
        json.dumps(chain_log, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print("[chain] all steps completed; summary at {}".format(
        OUTPUT_ROOT / "E039-E041_chain.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
