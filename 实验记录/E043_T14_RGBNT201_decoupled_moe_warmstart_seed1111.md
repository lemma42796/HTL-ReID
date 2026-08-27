# E043｜T14 七路解耦MoE warm start单次冲线

- 状态：运行准备中
- 登记时间：2026-08-28
- 对应论文实验：T14-DEMO-LITE
- 实验目的：从E042最佳mAP checkpoint热启动，用DeMo式异构特征分解与动态专家融合替代继续堆叠辅助损失，尝试单次达到DeMo*在RGBNT201上的73.7% mAP、80.5% Rank-1数值线。
- 初始化：`/root/autodl-tmp/outputs/HTL-ReID/E042_T12_peak_warmstart_seed1111/HTL-ReID_best.pth`；只加载模型权重，optimizer和scheduler重新初始化；新增七路MoE随机初始化。
- 唯一方案变化：保留E042的K=1 SFTS、三轮FACR、Part分支与原始CLS描述符；新增RGB/NIR/TIR三条模态专属路由、RGB-NIR/RGB-TIR/NIR-TIR三条两两共享路由和一条三模态共享路由，经多头动态门控形成七路MoE描述符。关闭E042的重建、异模态Triplet与BCC辅助项，不增加新的损失类型；MoE分支复用标准label-smoothed CE + Triplet。
- 配置：`configs/RGBNT201/paper/base.yml` + `configs/RGBNT201/fusion/t14_decoupled_moe_warmstart.yml`。
- 数据集与评估：RGBNT201 `train_171`训练、`test`评估；256×128；每epoch验证；关闭re-ranking。
- Seed / batch / epoch：1111 / 64 / 50；scheduler horizon 50 epoch；关闭严格确定性数值路径。
- Optimizer / LR / weight decay：Adam；基础LR 1e-4；backbone LR 1e-5；已有非backbone模块LR 1e-4；新增七路MoE及其BN/分类头LR 3.5e-4；weight decay 1e-4；10 epoch warm-up后cosine。
- 推理描述符：归一化FACR + 0.5倍原始三模态CLS + 0.3倍Part + 0.5倍七路MoE。
- 正式命令：`/root/miniconda3/bin/python tools/run_rgbnt201_fusion.py --single-experiment E043 --single-row T14-DEMO-LITE --single-config configs/RGBNT201/fusion/t14_decoupled_moe_warmstart.yml --single-output-name E043_T14_demo_lite_moe_seed1111 --seed 1111 --expected-train-epochs 50 --expected-max-epochs 50 --expected-base-lr 0.0001 --expected-batch-size 64 --expected-backbone-lr-factor 0.1 --expected-warmup-iters 10 --expected-resume-path /root/autodl-tmp/outputs/HTL-ReID/E042_T12_peak_warmstart_seed1111/HTL-ReID_best.pth --expected-strict-determinism 0`；runner内部强制`timeout --signal=TERM --kill-after=10s 30m`。
- 输出目录：`/root/autodl-tmp/outputs/HTL-ReID/E043_T14_demo_lite_moe_seed1111`。
- 预期产物：`resolved_config.yml`、`commit.txt`、`command.txt`、`stdout.log`、`train_log.txt`、`run_result.json`、`DONE`、TensorBoard事件、最佳mAP checkpoint、最佳Rank-1 checkpoint；若同一epoch达到目标则另存目标checkpoint。
- 判断口径：任一epoch同一模型同时达到或超过73.7% mAP与80.5% Rank-1即为本次单次冲线成功。
