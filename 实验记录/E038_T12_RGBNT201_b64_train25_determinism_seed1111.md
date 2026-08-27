# E038｜T12 batch64 train25 严格确定性基线

- 状态：运行中
- 登记时间：2026-08-27 21:55 CST
- 训练代码commit：待回填（远端HEAD，预期`46395b1`）
- 对应实验：T12-B64-DET25
- 实验目的：在E037验证通过的严格确定性机制上，建立batch 64、train25/max50的首个可重复正式基线；与E036（同配置、修复前非确定性运行）对比，确认确定性修复后的真实性能位置。
- 配置：`configs/RGBNT201/paper/base.yml` + `configs/RGBNT201/fusion/t12_sfts_k1_shared_token_recon_b64_25e.yml`
- 数据集与协议：RGBNT201 `train_171`训练、`test`评估；seed 1111；batch 64；每身份实例数8；每epoch验证；关闭re-ranking；输入256×128。
- 训练长度：`TRAIN_EPOCHS=25`，`MAX_EPOCHS=50`；scheduler horizon保持50。
- Optimizer / LR：Adam；base LR 3.5e-4；backbone factor 0.8（实际2.8e-4）；new module factor 1.0；10-epoch warm-up。
- 确定性：train_net.py全局启用`torch.use_deterministic_algorithms(True)`、cuDNN deterministic、关闭benchmark；runner设置`PYTHONHASHSEED=1111`与`CUBLAS_WORKSPACE_CONFIG=:4096:8`；每epoch写`determinism_trace.json`。
- Checkpoint：保存最佳checkpoint（沿用该配置的默认行为）。
- 输出目录：`/root/autodl-tmp/outputs/HTL-ReID/E038_T12_b64_train25_determinism_seed1111`
- 预期产物：`resolved_config.yml`、`command.txt`、`stdout.log`、`run_result.json`、`DONE`/`FAILED`、`determinism_trace.json`、最佳checkpoint、TensorBoard事件。
- 执行命令：`/root/miniconda3/bin/python tools/run_rgbnt201_fusion.py --single-experiment E038 --single-row T12-B64-DET25 --single-config configs/RGBNT201/fusion/t12_sfts_k1_shared_token_recon_b64_25e.yml --single-output-name E038_T12_b64_train25_determinism_seed1111 --seed 1111 --expected-train-epochs 25 --expected-base-lr 0.00035 --expected-batch-size 64`，runner内部强制`timeout --signal=TERM --kill-after=10s 30m`。
- 运行结果：待回填。
- 结论：待回填。
