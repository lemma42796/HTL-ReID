# E044｜E043最佳Rank-1 checkpoint描述符权重搜索

- 状态：运行准备中
- 登记时间：2026-08-28
- 实验目的：不重新训练，从E043最接近DeMo*的epoch 31 checkpoint出发，通过推理描述符权重与水平翻转TTA搜索补足0.48个百分点Rank-1差距。
- checkpoint：`/root/autodl-tmp/outputs/HTL-ReID/E043_T14_demo_lite_moe_seed1111/HTL-ReID_best_rank1.pth`，对应74.33% mAP、80.02% Rank-1。
- 数据集与评估：RGBNT201 `test`；256×128；seed 1111；关闭re-ranking；远端CUDA推理。
- 搜索空间：FACR权重固定1.0；原始CLS为0.25/0.5/0.75；Part为0/0.15/0.3；七路MoE为0.25/0.5/0.75/1.0，共36组。先评估原图特征，再评估按分量归一化后平均原图与水平翻转图的TTA特征，共72个候选。
- 实现：每张图的FACR、原始CLS、Part和MoE分量各提取一次；各分量先L2归一化，利用加权分块描述符的距离可分解性组合距离矩阵，避免为每组权重重复执行backbone。
- 正式命令：`timeout --signal=TERM --kill-after=10s 30m /root/miniconda3/bin/python tools/sweep_descriptor_weights.py --config-file configs/RGBNT201/paper/base.yml --config-file configs/RGBNT201/fusion/t14_decoupled_moe_warmstart.yml --checkpoint /root/autodl-tmp/outputs/HTL-ReID/E043_T14_demo_lite_moe_seed1111/HTL-ReID_best_rank1.pth --output-dir /root/autodl-tmp/outputs/HTL-ReID/E044_T14_descriptor_weight_sweep --tta-flip`。
- 输出目录：`/root/autodl-tmp/outputs/HTL-ReID/E044_T14_descriptor_weight_sweep`；runner日志为同级的`E044_T14_descriptor_weight_sweep.runner.log`；预期产物为resolved config、commit、command、完整72组结果及DONE标记。
- 判断口径：任一组合同时达到或超过73.7% mAP与80.5% Rank-1，即判定本次单次冲线成功。
