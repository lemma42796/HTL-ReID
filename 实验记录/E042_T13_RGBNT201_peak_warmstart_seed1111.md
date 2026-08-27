# E042｜T13 E031 warm start 单次峰值探索

- 状态：运行准备中
- 登记时间：2026-08-28
- 对应论文实验：T13-PEAK
- 实验目的：在用户明确以单次峰值为目标的前提下，从E031历史最佳checkpoint初始化，通过联合增强损失与推理描述符，尝试达到或超过DeMo*在RGBNT201上的73.7% mAP、80.5% Rank-1数值线。
- 初始化：`/root/autodl-tmp/outputs/HTL-ReID/E031_T12_sfts_k1_shared_token_recon_256x128_seed1111/HTL-ReID_best.pth`；只加载模型权重，optimizer、scheduler和新增Part Branch重新初始化。
- 模型变化：保留K=1 SFTS共享mask与三轮FACR；开启三条纹Part Branch；推理描述符拼接归一化FACR、0.5倍原始三模态CLS和0.3倍Part特征。
- 损失变化：FACR与原始CLS分支均使用等权label-smoothed CE + soft-margin Triplet；Part分支权重0.25；训练期同时重建RGB/NIR/TIR三个目标，使用归一化余弦损失加0.1倍Smooth L1且总权重0.2；增加权重0.3、margin 0.3的有监督跨模态batch-hard Triplet；增加权重0.05的BCC。
- 配置：`configs/RGBNT201/paper/base.yml` + `configs/RGBNT201/fusion/t13_peak_warmstart.yml`。
- 数据集与评估：RGBNT201 `train_171`训练、`test`评估；256×128；每epoch验证；关闭re-ranking。
- Seed / batch / epoch：1111 / 64 / 15；scheduler horizon 15 epoch；关闭严格确定性数值路径。
- Optimizer / LR / weight decay：Adam；基础LR 1e-4；backbone factor 0.1即1e-5；新增模块LR 1e-4；weight decay 1e-4；1 epoch warm-up后cosine。
- 计划命令：`/root/miniconda3/bin/python tools/run_rgbnt201_fusion.py --single-experiment E042 --single-row T13-PEAK --single-config configs/RGBNT201/fusion/t13_peak_warmstart.yml --single-output-name E042_T12_peak_warmstart_seed1111 --seed 1111 --expected-train-epochs 15 --expected-max-epochs 15 --expected-base-lr 0.0001 --expected-batch-size 64 --expected-backbone-lr-factor 0.1 --expected-warmup-iters 1 --expected-resume-path /root/autodl-tmp/outputs/HTL-ReID/E031_T12_sfts_k1_shared_token_recon_256x128_seed1111/HTL-ReID_best.pth --expected-strict-determinism 0`；runner内部强制`timeout --signal=TERM --kill-after=10s 30m`。
- 输出目录：`/root/autodl-tmp/outputs/HTL-ReID/E042_T12_peak_warmstart_seed1111`。
- 预期产物：`resolved_config.yml`、`commit.txt`、`command.txt`、`stdout.log`、`train_log.txt`、`run_result.json`、`DONE`、TensorBoard事件、mAP最佳checkpoint、Rank-1最佳checkpoint；若同一epoch同时达到73.7/80.5，另存目标checkpoint。
- 判断口径：只要任一epoch同一模型同时达到或超过73.7% mAP与80.5% Rank-1，即判定本次单次冲线成功；否则记录最高mAP及其Rank-1，并保留最高Rank-1 checkpoint。
