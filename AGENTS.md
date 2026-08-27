# HTL-ReID Agent Instructions

## Execution boundary

- Formal training runs, long-running dependency installations, dataset downloads, and large file transfers may be started when the user explicitly requests them.
- All model-executing tests, including smoke tests, training, validation, evaluation, inference, profiling, and controlled comparisons, must run on a CUDA GPU. Do not silently fall back to CPU; fail clearly when CUDA is unavailable. CPU is permitted only for static checks that do not execute the model, such as configuration parsing, syntax checks, and file inspection.
- Every training process has a hard wall-clock limit of 30 minutes, including the currently active run. Launch future training with `timeout --signal=TERM --kill-after=10s 30m ...`; never start an uncapped training process.
- Controlled comparisons must use the same fixed epoch count that fits within the 30-minute cap. Do not let different models train for different epoch counts merely because their throughput differs.
- Do not continuously watch or frequently poll long-running work. Prefer a detached/background process with output redirected to a log, perform at most one short startup check, then report the command, PID or job identifier, and log/artifact path and return control to the user. Monitor again only when the user explicitly asks for a status update.
- Read-only diagnostics, configuration inspection, dependency checks, and short smoke tests are allowed when they do not create large artifacts.
- Before proposing a formal run, state the config files, dataset, seed, batch size, epoch count, re-ranking setting, output directory, and expected artifacts.
- Never put credentials, passwords, access tokens, or private keys in commands saved to the repository, logs, or documentation.

## Remote storage budget

- Treat the remote system disk as limited to 30 GB and `/root/autodl-tmp` as limited to 50 GB.
- Store the repository, datasets, pretrained weights, checkpoints, outputs, and other persistent experiment artifacts under `/root/autodl-tmp`.
- Do not create duplicate datasets, extracted archive copies, extra environments, redundant checkpoints, or unneeded caches.
- Prefer `pip install --no-cache-dir ...` when giving installation commands.
- Before any large write, first inspect free space with `df -h` and inspect relevant directory sizes with `du -sh`.
- Keep only artifacts required for reproducibility: the resolved config, log, metrics/result file, and the best or explicitly required checkpoint. Periodic checkpoints remain disabled unless the user requests them.
- Do not delete checkpoints, datasets, logs, or other material artifacts without explicit user approval. Identify cleanup candidates and their sizes first.

## Verified remote baseline

- Expected GPU: NVIDIA GeForce RTX 5090 with 32 GB VRAM.
- Expected runtime: Python 3.12, PyTorch 2.8.0 with CUDA 12.8, and torchvision 0.23.0.
- Remote repository path: `/root/autodl-tmp/HTL-ReID`.
- Remote SSH alias: `autodl-reid`. Use the alias instead of embedding the current endpoint in commands.
- Dataset root: `/root/autodl-tmp/datasets`.
- Pretrained-weight path: `/root/autodl-tmp/pretrained/vit_base_patch16_224_augreg2_in21k_ft_in1k.pth`.
- Output root: `/root/autodl-tmp/outputs/HTL-ReID`.
- In non-interactive SSH commands, use `/root/miniconda3/bin/python` if `python` is not on `PATH`.

## Remote network policy

- GitHub direct access from AutoDL is intermittent. For all remote Git operations that access GitHub, enable AutoDL academic acceleration by default only inside a temporary subshell, for example: `( source /etc/network_turbo && git pull --ff-only )`.
- Do not first attempt direct GitHub access from AutoDL unless diagnosing the accelerator itself. Do not rewrite Git remotes to a `ghproxy` URL, set a persistent global Git proxy, or source `/etc/network_turbo` from shell startup files.
- Let the temporary subshell end immediately after the GitHub Git command so proxy variables cannot affect pip, dataset access, model or dataset downloads, or unrelated network traffic. Do not use academic acceleration for those other operations unless the user explicitly requests it for a specific command.
- This temporary-subshell policy was verified on 2026-08-26 with accelerated `git ls-remote` returning commit `bc4e0bb` and accelerated `git pull --ff-only` updating the training machine to `32bbc7b`.

## Experiment discipline

- Follow the current paper plan and freeze the paper configs before formal training.
- Use the same backbone, pretrained weights, input size, sampler, batch size, optimizer, schedule, seed policy, and evaluation protocol for controlled M0-M3 comparisons.
- Main-paper evaluation must explicitly disable re-ranking; report re-ranking only as a separately labeled result.
- Register every formal or failed run in `实验记录.md`, and store its complete command, commit, config, seed, paths, metrics, and conclusion in a dedicated `实验记录/E*.md` file.

## Documentation discipline

- Do not record session-by-session actions, troubleshooting narratives, transfer progress, or other operational流水账 in project documents.
- Keep `AGENTS.md` limited to stable rules, constraints, and path conventions.
- Keep `项目状态与TODO.md` limited to the latest valid state, unresolved issues, and current tasks; replace stale information instead of appending history.
- Keep `论文大修执行方案.md` limited to durable research and revision decisions.
- Use `实验记录.md` only as the durable experiment index and result summary. Use `实验记录/E*.md` only for reproducible formal or failed experiment evidence that must be retained long term.
