# E037｜T12严格确定性CUDA配对验证

- 状态：待启动
- 登记时间：2026-08-27 18:15 CST
- 对应实验：DET-A / DET-B
- 实验目的：修复E031/E036同seed曲线分叉问题，并用两次完全相同的CUDA训练验证采样顺序、T12重建目标、loss和评估指标可重复。
- 代码变化：关闭cuDNN benchmark并启用严格确定性算法；在CUDA初始化前设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`；runner对子进程设置`PYTHONHASHSEED=1111`；DataLoader显式固定generator和worker的Python/NumPy/Torch seed；`RandomIdentitySampler`使用`seed+epoch`独立随机源；T12重建目标使用独立固定seed生成器；每epoch写入`determinism_trace.json`。
- 配置：`configs/RGBNT201/paper/base.yml` + `configs/RGBNT201/fusion/t12_sfts_k1_shared_token_recon_b64_determinism_smoke2.yml`
- 数据集与协议：RGBNT201 `train_171`训练、`test`评估；seed 1111；batch 64；每身份实例数8；每epoch验证；关闭re-ranking。
- 训练长度：DET-A与DET-B均为2 epoch，`MAX_EPOCHS=50`；两次受控比较使用完全相同的固定epoch数。
- Optimizer / LR：Adam；base/new-module LR 3.5e-4；backbone factor 0.8，实际LR 2.8e-4；10-epoch warm-up与50-epoch cosine horizon保持不变。
- Checkpoint：关闭保存，避免两次短测试产生约864 MB无必要权重；保留解析配置、命令、stdout、结果JSON、DONE和确定性trace。
- DET-A输出：`/root/autodl-tmp/outputs/HTL-ReID/E037A_T12_determinism_b64_smoke2_seed1111`
- DET-B输出：`/root/autodl-tmp/outputs/HTL-ReID/E037B_T12_determinism_b64_smoke2_seed1111`
- 计划命令A：`/root/miniconda3/bin/python tools/run_rgbnt201_fusion.py --single-experiment E037-A --single-row DET-A --single-config configs/RGBNT201/fusion/t12_sfts_k1_shared_token_recon_b64_determinism_smoke2.yml --single-output-name E037A_T12_determinism_b64_smoke2_seed1111 --seed 1111 --expected-train-epochs 2 --expected-base-lr 0.00035 --expected-batch-size 64`。
- 计划命令B：与A相同，仅实验/row/output分别改为`E037-B`、`DET-B`和`E037B_T12_determinism_b64_smoke2_seed1111`。两次runner内部均强制`timeout --signal=TERM --kill-after=10s 30m`。
- 判断口径：两次均须在CUDA上returncode 0且无确定性算法报错；两个`determinism_trace.json`除输出无关信息外必须字节一致，逐epoch采样SHA256、完整重建目标序列、train loss/accuracy和mAP/Rank指标全部一致。任一项不一致即未修复。
- 后续：通过后再注册batch 64、train25/max50严格确定性基线；本测试不用于论文精度结果，不触发消融。
