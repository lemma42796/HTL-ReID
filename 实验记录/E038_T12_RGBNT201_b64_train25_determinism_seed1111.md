# E038｜T12 batch64 train25 严格确定性基线

- 状态：已完成（严格确定性机制正常运行；主指标明显低于同配置修复前的E036）
- 登记时间：2026-08-27 21:55 CST；完成时间：2026-08-27 18:57 CST（远端时间）
- 训练代码commit：`46395b1ce4f2ee0c858d14a1cc9c3f0fe6760ace`
- 对应实验：T12-B64-DET25
- 实验目的：在E037校验通过的严格确定性机制上，建立batch 64、train25/max50的首个严格确定性正式基线；与E036（同配置、修复前非确定性运行）对比，估计固定训练轨迹后的性能位置。本实验不是对E031最高性能的复现。
- 配置：`configs/RGBNT201/paper/base.yml` + `configs/RGBNT201/fusion/t12_sfts_k1_shared_token_recon_b64_25e.yml`
- 数据集与协议：RGBNT201 `train_171`训练、`test`评估；seed 1111；batch 64；每身份实例数8；每epoch验证；关闭re-ranking；输入256×128。
- 训练长度：`TRAIN_EPOCHS=25`，`MAX_EPOCHS=50`；scheduler horizon保持50。
- Optimizer / LR：Adam；base LR 3.5e-4；backbone factor 0.8（实际2.8e-4）；new module factor 1.0；10-epoch warm-up。
- 确定性：train_net.py全局启用`torch.use_deterministic_algorithms(True)`、cuDNN deterministic、关闭benchmark；runner设置`PYTHONHASHSEED=1111`与`CUBLAS_WORKSPACE_CONFIG=:4096:8`；每epoch写`determinism_trace.json`。
- Checkpoint：保存最佳checkpoint（沿用该配置的默认行为）。
- 输出目录：`/root/autodl-tmp/outputs/HTL-ReID/E038_T12_b64_train25_determinism_seed1111`
- 预期产物：`resolved_config.yml`、`command.txt`、`stdout.log`、`run_result.json`、`DONE`/`FAILED`、`determinism_trace.json`、最佳checkpoint、TensorBoard事件。
- 执行命令：`/root/miniconda3/bin/python tools/run_rgbnt201_fusion.py --single-experiment E038 --single-row T12-B64-DET25 --single-config configs/RGBNT201/fusion/t12_sfts_k1_shared_token_recon_b64_25e.yml --single-output-name E038_T12_b64_train25_determinism_seed1111 --seed 1111 --expected-train-epochs 25 --expected-base-lr 0.00035 --expected-batch-size 64`，runner内部强制`timeout --signal=TERM --kill-after=10s 30m`。
- 运行结果：returncode 0，耗时500.1秒；最佳epoch 20，62.05 mAP、65.79 Rank-1、77.63 Rank-5、82.06 Rank-10。平均约0.217秒/batch（54 iter/epoch）；`determinism_trace.json` SHA256为`6fd68de2ee03e388541a2d75abfba7d76843f022ed0159c4eb11692ffa2d97fc`，25个epoch的采样与重建trace全部正常写入；结果JSON、DONE、解析配置、命令、日志、确定性trace和约431 MB最佳checkpoint均已保留，训练进程已退出，无OOM、NaN或timeout。
- 逐epoch曲线（mAP）：epoch 1–7为32.56/52.25/58.12/61.36/58.21/59.58/58.70；随后进入高波动区间，epoch 8–13跌至45–55，14–20回升至55–62（最佳62.05 @ epoch 20），21–25再度回落到51–60。epoch 2的52.25/54.07与E037 DET-A/B的2-epoch结果逐位一致，间接再次确认确定性机制。
- 对比：与同配置、同seed的修复前E036（最佳epoch 24，66.49 mAP、68.06 Rank-1）相比，主指标下降4.44/2.27个百分点；也低于E035（68.23/73.68）与E031（71.54/75.24）。E036与E038除确定性开关外配置一致，该差距不能归因于batch、LR或epoch。
- 结论：严格确定性机制工作正常，同一固定训练轨迹可以逐位重复，但E038没有复现E031最高性能；固定轨迹后的同配置性能明显下降且逐epoch验证曲线波动加剧（45–62区间）。因此：1) 修复前的旧结果（含E031/E036）不能直接作为严格确定性口径下的性能参考；2) E038只是该口径的首个单seed基线，不是历史最高性能的复现结果；3) 本实验当时留下的“是否恢复宽松数值路径”问题，已由后续E039–E041关闭，最终决定保持严格确定性并使用三seed均值。
