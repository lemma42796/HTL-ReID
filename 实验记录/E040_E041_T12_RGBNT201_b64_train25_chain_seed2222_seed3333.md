# E040/E041｜T12 batch64 train25 方案C链seed 2222/3333配对

- 状态：待E039归因结果决定模式后自动启动（由`tools/run_rgbnt201_determinism_chain.py`编排）
- 登记时间：2026-08-27 22:50 CST
- 训练代码commit：与E039同一提交
- 对应实验：T12-B64-CHAIN25（E040 seed 2222、E041 seed 3333）
- 实验目的：在E039归因决定的口径下补齐两个额外seed，与seed 1111（严格侧E038或宽松侧E039）组成三seed基线，重建可重复且接近旧水平的性能参考点。
- 模式决策：E039 mAP ≥ 64.05（严格基线62.05 + 2.0）时采用宽松配置`configs/RGBNT201/fusion/t12_sfts_k1_shared_token_recon_b64_25e_relaxed.yml`，否则采用严格配置`configs/RGBNT201/fusion/t12_sfts_k1_shared_token_recon_b64_25e.yml`。seed 1111对应结果分别取E039或E038。
- 配置：`configs/RGBNT201/paper/base.yml` + 上述模式配置；其余协议与E038完全一致（256×128、Adam、batch 64、train25/max50、10-epoch warm-up、关闭re-ranking、每epoch验证）。
- 数据集与协议：RGBNT201 `train_171`训练、`test`评估。
- 输出目录：`/root/autodl-tmp/outputs/HTL-ReID/E040_T12_b64_train25_{strict|relaxed}_seed2222`、`E041_T12_b64_train25_{strict|relaxed}_seed3333`（按决策模式填充）。
- 预期产物：每次运行含`resolved_config.yml`、`command.txt`、`stdout.log`、`run_result.json`、`DONE`/`FAILED`、`determinism_trace.json`、最佳checkpoint、TensorBoard事件；链级汇总写入`/root/autodl-tmp/outputs/HTL-ReID/E039-E041_chain.json`。
- 运行结果：待回填。
- 结论：待回填。
