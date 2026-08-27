# E039｜T12 batch64 train25 确定性代价归因（宽松数值路径）

- 状态：已启动（方案C链第1步，由`tools/run_rgbnt201_determinism_chain.py`编排）
- 登记时间：2026-08-27 22:50 CST
- 训练代码commit：待回填（远端HEAD，预期为登记本次代码变更后的提交）
- 对应实验：T12-B64-RELAX25
- 实验目的：E038（严格确定性）较同配置修复前E036下降4.44/2.27。本实验保留修复引入的固定采样器、DataLoader worker种子和T12重建目标生成器，但将`SOLVER.STRICT_DETERMINISM`置0，恢复修复前的cuDNN/cuBLAS数值行为（`use_deterministic_algorithms`关闭、cuDNN benchmark开启），用于隔离掉点来源：若结果回到66附近，代价主要来自确定性算子；若仍在62附近，代价来自采样轨迹差异。
- 配置：`configs/RGBNT201/paper/base.yml` + `configs/RGBNT201/fusion/t12_sfts_k1_shared_token_recon_b64_25e_relaxed.yml`（与E038配置逐键一致，仅新增`STRICT_DETERMINISM: 0`）。
- 数据集与协议：RGBNT201 `train_171`训练、`test`评估；seed 1111；batch 64；每身份实例数8；每epoch验证；关闭re-ranking；输入256×128。
- 训练长度：`TRAIN_EPOCHS=25`，`MAX_EPOCHS=50`；scheduler horizon保持50。
- Optimizer / LR：Adam；base LR 3.5e-4；backbone factor 0.8（实际2.8e-4）；new module factor 1.0；10-epoch warm-up。
- 确定性：不启用`use_deterministic_algorithms`与cuBLAS确定性workspace（runner环境变量仍存在但不生效）；采样顺序、重建目标仍由固定种子生成器驱动；`determinism_trace.json`仍逐epoch写入，但逐位可复现性不再被要求。
- Checkpoint：保存最佳checkpoint（沿用该配置的默认行为）。
- 输出目录：`/root/autodl-tmp/outputs/HTL-ReID/E039_T12_b64_train25_relaxed_seed1111`
- 预期产物：`resolved_config.yml`、`command.txt`、`stdout.log`、`run_result.json`、`DONE`/`FAILED`、`determinism_trace.json`、最佳checkpoint、TensorBoard事件。
- 决策规则：E039 mAP ≥ 62.05 + 2.0 时，后续E040/E041采用宽松模式；否则保持严格确定性模式。规则固化在链式脚本中。
- 运行结果：待回填。
- 结论：待回填。
