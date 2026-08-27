# E040/E041｜T12 batch64 train25 方案C链seed 2222/3333配对

- 状态：已完成（E039未达门槛，链脚本自动选定严格模式）
- 登记时间：2026-08-27 22:50 CST；完成时间：2026-08-27（远端）
- 训练代码commit：`5087d9cf61cb8906ad8439eb26016a63e1d81294`（与E039同一提交）
- 对应实验：T12-B64-CHAIN25（E040 seed 2222、E041 seed 3333）
- 实验目的：在E039归因决定的口径下补齐两个额外seed，与seed 1111（严格侧E038或宽松侧E039）组成三seed基线，重建可重复且接近旧水平的性能参考点。
- 模式决策：E039 mAP 62.11 < 64.05，未达门槛，链脚本自动选定严格配置`configs/RGBNT201/fusion/t12_sfts_k1_shared_token_recon_b64_25e.yml`；seed 1111对应结果取E038（62.05/65.79）。
- 配置：`configs/RGBNT201/paper/base.yml` + 上述模式配置；其余协议与E038完全一致（256×128、Adam、batch 64、train25/max50、10-epoch warm-up、关闭re-ranking、每epoch验证）。
- 数据集与协议：RGBNT201 `train_171`训练、`test`评估。
- 输出目录：`/root/autodl-tmp/outputs/HTL-ReID/E040_T12_b64_train25_strict_seed2222`、`/root/autodl-tmp/outputs/HTL-ReID/E041_T12_b64_train25_strict_seed3333`。
- 预期产物：每次运行含`resolved_config.yml`、`command.txt`、`stdout.log`、`run_result.json`、`DONE`/`FAILED`、`determinism_trace.json`、最佳checkpoint、TensorBoard事件；链级汇总写入`/root/autodl-tmp/outputs/HTL-ReID/E039-E041_chain.json`。
- 运行结果：E040 returncode 0，耗时521.7秒；最佳epoch 4，69.59 mAP、71.29 Rank-1、82.54 Rank-5、89.59 Rank-10。E041 returncode 0，耗时515.5秒；最佳epoch 24，63.51 mAP、64.83 Rank-1、76.91 Rank-5、83.97 Rank-10。产物齐全，无OOM、NaN或timeout。
- 逐epoch要点：E040曲线震荡剧烈（45–70），最佳69.59出现在epoch 4，epoch 15的Rank-1更高（72.37）但mAP 69.54；最佳epoch过早提示曲线噪声大、最佳点选择敏感。E041在33–64区间震荡，后期缓慢爬升，最佳63.51 @ epoch 24。
- 三seed严格确定性汇总（seed 1111/2222/3333 = E038/E040/E041）：mAP 62.05/69.59/63.51，均值约65.05、样本标准差约4.0；Rank-1 65.79/71.29/64.83，均值约67.30、样本标准差约3.5。
- 结论：E040单个seed达到69.59/71.29，说明严格确定性机制不会必然把所有训练限制在E038的低点，但它仍未复现E031的71.54/75.24。三seed均值只有65.05/67.30，样本标准差约4.0/3.5，且最佳epoch分布极不稳定（4/20/24）；因此不能把E040或任何单次高点称为最高性能复现。后续口径保持严格确定性，性能表述必须使用三seed均值；曲线震荡与最佳点选择策略仍待解决。
