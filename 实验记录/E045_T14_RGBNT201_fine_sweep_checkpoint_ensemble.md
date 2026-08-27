# E045｜E043细粒度描述符、TTA与checkpoint集成榨取

- 状态：运行准备中
- 登记时间：2026-08-28
- 实验目的：在E044已超过DeMo*的基础上，不重新训练，继续搜索描述符与原图/翻转融合比例，并利用epoch 31/33 checkpoint soup和距离集成提高单次峰值。
- checkpoint：E043 `HTL-ReID_best_rank1.pth`（epoch 31，74.33/80.02）与`HTL-ReID_best.pth`（epoch 33，74.46/79.67）。
- 数据集与评估：RGBNT201 `test`；256×128；seed 1111；batch 64；关闭re-ranking；远端CUDA推理。
- 搜索策略：Part固定关闭；FACR作为单位权重。每个模型先对翻转融合alpha=0/0.25/0.5/0.75/1、CLS权重0.2至1.0步长0.1、MoE权重0.5至1.5步长0.1粗搜，再分别围绕最佳Rank-1和最佳mAP以alpha步长0.05、描述符权重步长0.02局部细搜。
- checkpoint利用：分别搜索epoch 31与33；再搜索两者0.25/0.5/0.75权重soup，并对最佳soup细搜；最后对各单模型最佳距离矩阵以0.025步长做两两集成。
- 正式命令：`timeout --signal=TERM --kill-after=10s 30m /root/miniconda3/bin/python tools/sweep_checkpoint_ensemble.py --config-file configs/RGBNT201/paper/base.yml --config-file configs/RGBNT201/fusion/t14_decoupled_moe_warmstart.yml --rank1-checkpoint /root/autodl-tmp/outputs/HTL-ReID/E043_T14_demo_lite_moe_seed1111/HTL-ReID_best_rank1.pth --map-checkpoint /root/autodl-tmp/outputs/HTL-ReID/E043_T14_demo_lite_moe_seed1111/HTL-ReID_best.pth --output-dir /root/autodl-tmp/outputs/HTL-ReID/E045_T14_fine_sweep_ensemble`。
- 输出目录：`/root/autodl-tmp/outputs/HTL-ReID/E045_T14_fine_sweep_ensemble`；runner日志为同级`E045_T14_fine_sweep_ensemble.runner.log`；预期产物为resolved config、commit、command、完整搜索结果、DONE及命中时的最佳单模型soup。
- 判断口径：保持mAP不低于73.7%，优先最大化Rank-1；记录最大mAP候选作为补充。
