# E013｜RGBNT201 T6 checkpoint固定K筛选

- 状态：已完成
- 日期：2026-08-25 CST
- 实验目的：利用E012共享权重checkpoint快速筛选固定`K∈{1,2,4,8,16}`，仅将最佳候选送入独立正式训练。
- 数据集与协议：RGBNT201 test；三模态联合评估；seed 1111；关闭re-ranking。
- 配置：`configs/RGBNT201/paper/base.yml` + `configs/RGBNT201/fusion/t6_sfts_learnable_k_facr.yml`
- Checkpoint：`/root/autodl-tmp/outputs/HTL-ReID/E012_T6_sfts_learnable_k_facr_seed1111/HTL-ReID_best.pth`
- 评估脚本：`tools/eval_sfts_k_sweep.py`
- 结果文件：`/root/autodl-tmp/outputs/HTL-ReID/E013_T6_fixed_k_sweep/results.json`
- Re-ranking：关闭
- 结果（mAP / Rank-1）：K=1 59.6751/59.4498；K=2 59.8778/59.9282；K=4 60.0353/60.1675；K=8 60.0675/59.6890；K=16 60.0716/59.8086。
- 方法限制：checkpoint训练时随机切换K，权重并非针对任何固定K独立优化；K=4/8/16的代理差距不超过0.0363 mAP，不能证明K=16显著更优；筛选直接读取test指标，存在测试集选参风险，不得作为论文中的严格最优K搜索。
- 结论：按代理mAP选择K=16进入E014独立训练；本筛选仅用于节省计算的候选排序，不能替代独立训练，也不得声称K=16是严格搜索得到的最优值。
