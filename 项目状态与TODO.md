# HTL-ReID 项目状态与 TODO

> 本文档只保留当前有效状态和未完成事项，不记录会话操作流水。正式实验事实见《实验记录.md》及`实验记录/E*.md`。

## 一、信息入口

- 最后更新：2026-08-27 02:10 CST
- 项目根目录：`/Users/a123/Documents/reid`
- 代码目录：`/Users/a123/Documents/reid/HTL-ReID`
- 远端代码目录：`/root/autodl-tmp/HTL-ReID`
- 远端登录：`ssh autodl-reid`
- SSH别名：`autodl-reid`与`autodl-reid-new`均指向当前新训练机；`autodl-reid-old`保留原机器配置，但最近一次检查不可连接
- 数据集根目录：`/root/autodl-tmp/datasets`
- 输出根目录：`/root/autodl-tmp/outputs/HTL-ReID`
- 实验索引：`/Users/a123/Documents/reid/HTL-ReID/实验记录.md`

## 二、当前研究目标

M0–M3、T1–T12、固定K参数敏感性及三seed T2/K1配对实验已经完成。T11的seed 1111大幅提升未在seed 2222下复现。T12在不改变K1/FACR推理路径的前提下，增加共享、目标模态条件化的训练期token重建头；E029/E030在seed 1111/2222均同时超过配对K1，两次分别提升2.51/0.59和2.01/1.08 mAP/Rank-1，因此T12作为当前优先结构，K1保留为冻结配对基线。基于时间成本，不再追加RGBNT201 seed 3333，后续优先转向跨数据集验证与论文工作。

当前融合实验：

| 行 | 方法 | 目的 |
|---|---|---|
| T1 / E006 | M0 + 纯TPM | 正确复现TOP-ReID的TPM循环，作为明确引用的对照 |
| T2 / E007 | M0 + 自适应全连接路由 | 隔离验证动态跨模态路由，不使用FACSS分数 |
| T3 / E008 | HS/FACSS连续评分 + FACR | 验证token重要性软引导跨模态融合是否超过T1、T2和M2 |
| T3R / E009 | 零初始化有界FACSS评分 + FACR | 修复强评分先验，重新验证FACSS软引导 |
| T4 / E010 | FACSS Top-K硬选择 + FACR | 检验真正的选择后融合 |
| T5–T7 / E011–E018 | SFTS + FACR及固定K敏感性 | 验证SFTS硬选择和K值；K=1与K=16进入候选 |
| T8 / E019 | K=16 + FACR路由均衡损失 | 已否定：原路由没有明显来源塌缩 |
| T9 / E020 | 全tokens FACR + 最终自身patch细化 | 已否定：无条件自身信息重读明显退化 |
| T10 / E021、E024 | K=1 SFTS丢弃信息摘要 + masked FACR自细化 | 额外seed未复现优势；不作为最终精度候选 |
| T2/K1三seed / E007、E015、E022、E023、E025、E026 | 全tokens T2 vs 固定K=1 SFTS + FACR | K1平均mAP略高且Rank-1三次一致领先；锁定K1 |
| T11 / E027、E028 | K1 + FACR前置独立masked aggregation | seed 1111大幅提升，但seed 2222的mAP/Rank-1均略低于K1；不替换最终K1 |
| T12 / E029、E030 | K1 + 训练期共享跨模态token重建 | 两个seed均同时超过配对K1；停止同数据集追加seed，转跨数据集 |

TPM只作为TOP-ReID引用复现，不能改名冒充原创。FACR的实质差异是固定循环变为样本自适应全连接路由；FACSS连续分数、T10残差摘要和自细化均已被稳定性实验排除出最终主线。

### 参考论文来源与创新边界

当前结构是在`/Users/a123/Documents/reid/ReID相关论文`论文集中寻找解决方案后形成，来源必须按以下边界表述：

- `Magic Tokens: Select Diverse Tokens for Multi-modal Object Re-Identification`（EDITOR）提供SFTS的空间/频率token选择、逐head Top-K、跨head/跨模态共享并集，以及HMA“独立聚合后再协同聚合”的层级顺序；项目保留其硬mask选择规则。T11明确借鉴HMA的独立聚合顺序，但只增加共享权重的模态内masked aggregation，后续协同阶段仍为项目FACR，并未复现完整HMA、BCC或OCFR；论文必须引用EDITOR，不能把独立聚合本身宣称为原创；
- `TOP-ReID: Multi-spectral Object Re-Identification with Token Permutation`提供TPM的CLS读取其他模态patch并循环聚合的直接参考；T1是明确引用的模块复现，FACR则把固定循环改为同时读取两个来源并进行样本自适应路由。T12直接受TOP-ReID CRM的跨模态token重建监督启发，必须明确引用CRM；其工程差异是针对共享ViT使用单个共享、模态条件化的轻量预测器，每批只重建一个目标模态且不进入推理路径，不得将token重建思想宣称为原创；
- `DeMo: Decoupled Feature-Based Mixture of Experts for Multi-Modal Object Re-Identification`中的ATMoE与FACR都包含按样本动态加权多模态信息的思想，可作为相关工作和概念对照；但现有代码及实验记录没有FACR直接复现或改造ATMoE的证据，不得把DeMo写成FACR的直接代码来源；
- T10新增的“将SFTS丢弃patch压缩为每模态一个注意力加权残差摘要，并在FACR后做受限自模态细化”不是上述论文的原有模块，但E024未复现精度优势，因此只能作为项目探索与信息保真消融，不能写成最终贡献。SFTS、TPM、交叉注意力、门控和动态加权本身均不得宣称为原创。

## 三、当前方法定义

### 1. TPM对照

每个模态的CLS依次读取另外两个模态的完整patch tokens，最后读取自身patch tokens：

- RGB CLS：NIR patches → TIR patches → RGB patches；
- NIR CLS：TIR patches → RGB patches → NIR patches；
- TIR CLS：RGB patches → NIR patches → TIR patches。

每一步使用上一步更新后的CLS，三个最终CLS直接拼接并接受CE + Triplet监督。当前项目使用共享ViT，TOP-ReID原文使用三个独立ViT，因此T1是TPM模块对照，不是整套TOP-ReID复现。

### 2. 修改后的FACSS

FACSS保留旧Top-K二值mask接口，但T3主路径不在融合前删除token。它输出每个模态所有patch的连续重要性分数：

`FACSS score = 模态内层级注意力 + α × 跨模态余弦一致性`

- HS使用ViT第4、8、12层注意力形成层级候选；
- 跨模态一致性为每个token在其他模态token中寻找高余弦匹配；
- 连续分数用于FACR注意力偏置；
- 二值mask仅保留用于旧路径和消融。

T3设置`FACR_DETACH_SCORES=0`，使FACSS的α网络能从识别损失获得梯度；若detach且关闭旧0.15描述符分支，FACSS评分网络将无法训练。

### 3. FACR

FACR对每个目标模态执行：

1. T11可选地让每个模态CLS先读取自身被共享mask选中的patch；冻结K1关闭该步骤；
2. CLS分别对另外两个模态的完整patch tokens做交叉注意力；
3. 可选的FACSS连续分数经过零初始化可学习增益形成有界soft attention bias；T10关闭该路径并使用SFTS共享mask；
4. 样本自适应路由决定两个来源模态的相对权重；
5. 可学习逐通道门控控制跨模态上下文注入量；
6. 重复3轮后，可选最终自模态细化；T10只让目标CLS读取被SFTS丢弃patch的注意力加权摘要，不重复读取全量自身patch；
7. 拼接RGB/NIR/TIR三个CLS，直接接受CE + Triplet监督。

T2关闭FACSS分数，用于隔离路由本身；E008强对数偏置与E009有界偏置均未超过T2，评分引导已放弃。E020证明全量自身patch细化有害；E021的丢弃信息摘要单seed结果较好，但E024未复现并低于同seed T2/K1，因此该路径也已退出最终主线。

### 4. T12训练期共享跨模态token重建

T12保持K1的SFTS与FACR推理路径不变。训练时每个batch随机选择RGB/NIR/TIR中的一个目标模态，共享重建器只读取其他两个模态的完整patch tokens，结合源/目标模态embedding按对应patch位置预测目标tokens。目标tokens仅作为stop-gradient教师，损失为fp32归一化余弦距离；默认系数0.1，前5 epoch通过既有辅助损失warmup逐步加入。该头不参与识别特征生成，eval分支不调用，因此不增加推理计算。

## 四、代码与验证状态

- T12代码已提交并用于E029/E030正式训练，训练代码快照为`0daf4d2`；能力默认关闭，K1/T11配置与旧checkpoint结构不变。E020/E021训练时runner记录的HEAD仍为`7886101`，但其工作区快照与随后冻结的`bc4e0bb`一致。E006–E008训练代码快照为`1b2aa71`，E009训练代码快照为`e2af604`，E014训练代码快照为`df17584`，E015–E018训练代码快照为`2cc3438`。
- 核心提交：`0089f1b`（TPM/FACR/FACSS接口）、`157a99c`（CUDA测试入口）、`1b2aa71`（E006–E008顺序runner）、`e2af604`（零初始化有界评分偏置及单实验runner）、`7886101`（可选FACR批级路由均衡损失与统计）、`bc4e0bb`（SFTS丢弃信息残差摘要与FACR最终自细化）、`32bbc7b`（融合runner显式seed覆盖）、`a78c007`（FACR前置独立masked aggregation与T11）、`0daf4d2`（训练期共享跨模态token重建与T12）。
- 核心代码：`modeling/fusion_part/TPM.py`、`modeling/fusion_part/HS_FACSS.py`、`modeling/fusion_part/CrossModalReconstruction.py`、`modeling/make_model.py`。
- 配置：`configs/RGBNT201/fusion/t1_tpm.yml`、`t2_adaptive_routing.yml`、`t3_m2_facr.yml`、`t7_sfts_fixed_k16_facr.yml`、`t8_sfts_fixed_k16_route_balance.yml`、`t9_facr_self_refine.yml`、`t10_sfts_k1_residual_facr.yml`、`t11_sfts_k1_independent_facr.yml`、`t12_sfts_k1_shared_token_recon.yml`。
- T1–T12相关CPU前向、反向、mask不变性、T12目标隔离/stop-gradient/评估旁路与完整模型接线测试已通过；E029/E030均已在RTX 5090上完成20 epoch正式训练，无OOM、NaN或timeout。
- 计算优化已经完成：HS rollout降阶、跨模态相似度复用、动态Top-K与频域mask批量化、训练同步/传输及AdamW参数组优化。
- T8路由均衡能力已实现并保持默认关闭：`FACR_ROUTE_BALANCE_WEIGHT=0.0`不改变现有配置和checkpoint。E019以0.05正式训练后主指标退化，且路由诊断未发现明显塌缩，因此仅保留该能力用于诊断/消融，不进入最终模型。
- T9/T10能力已实现且默认关闭：SFTS可输出每模态一个丢弃patch残差摘要，FACR可选最终自模态细化。T9全tokens自细化和T10残差摘要路径均已否定为最终精度方案，仅保留代码用于复现消融。
- T12能力已实现且默认关闭：单个共享重建器每批随机选一个目标模态，目标tokens stop-gradient，仅其他两模态及重建头接受该辅助损失梯度；评估不调用重建头。

## 五、有效实验结果

| 实验 | 配置 | mAP | Rank-1 | 判断 |
|---|---|---:|---:|---|
| E001 / M0 | 共享ViT基线 | 62.45 | 62.80 | 基线 |
| E002 / M1 | M0 + HS | 62.73 | 63.64 | 小幅正增益 |
| E003 / M2 | M1 + FACSS | 63.50 | 64.83 | M系列最好；相对M1 +0.77/+1.19 |
| E004 / M3 | M2 + QAWF | 60.03 | 59.57 | 否定QAWF |
| E005 / L1 | 旧A2质量感知频域组合 | 61.61 | 62.08 | 不并入主线 |
| E006 / T1 | M0 + 纯TPM | 61.36 | 61.72 | 低于M0和M2，不支持直接采用纯TPM |
| E007 / T2 | M0 + 自适应全连接路由 | 63.82 | 65.07 | 全tokens主线；相对T1 +2.46/+3.35，相对M2 +0.32/+0.24 |
| E008 / T3 | HS/FACSS强对数评分 + FACR | 62.70 | 61.00 | 低于T2与M2；不保留强偏置实现 |
| E009 / T3R | 零初始化有界FACSS评分 + FACR | 62.70 | 64.23 | Rank-1较E008恢复，但仍低于T2；放弃评分引导 |
| E010 / T4 | FACSS固定Top-K=16硬选择 + FACR | 59.90 | 59.81 | 相对T2 -3.92/-5.26；否定融合前硬选择 |
| E011 / T5 | EDITOR原始SFTS硬选择 + FACR | 62.37 | 63.04 | 性能开销仅约1.5%，但相对T2 -1.45/-2.03 |
| E012 / T6 | 可学习K的SFTS + FACR | 60.04 | 60.17 | argmax K=4、最终保留27.10%；K搜索未充分收敛，不采用 |
| E014 / T7-K16 | 固定K=16的SFTS + FACR | 63.80 | 65.07 | 固定K组最高mAP；与T2基本持平 |
| E015 / K1 | 固定K=1的SFTS + FACR | 63.31 | 66.99 | seed 1111；与E023/E026组成最终三seed结果 |
| E016 / K2 | 固定K=2的SFTS + FACR | 62.35 | 64.59 | 实际保留17.38% |
| E017 / K4 | 固定K=4的SFTS + FACR | 62.83 | 64.71 | 实际保留26.71% |
| E018 / K8 | 固定K=8的SFTS + FACR | 62.18 | 63.76 | 实际保留41.64%；本组最低mAP |
| E019 / T8 | K=16 + FACR路由均衡损失 | 62.68 | 64.23 | 相对E014 -1.12/-0.84；否定该损失 |
| E020 / T9 | T2 + FACR最终自身patch细化 | 62.33 | 61.12 | 相对T2 -1.49/-3.95；否定无条件自细化 |
| E021 / T10 | K=1 + 丢弃信息残差摘要 + masked FACR自细化 | 63.88 | 66.27 | seed 1111探索结果；E024未复现，不采用 |
| E022 / T2-S1 | T2额外seed 2222 | 64.41 | 66.15 | T2第二次结果；正常完成 |
| E023 / K1-S1 | K1额外seed 2222 | 64.93 | 67.82 | 同seed最高mAP/Rank-1；实际保留11.63% |
| E024 / T10-S1 | T10额外seed 2222 | 63.62 | 65.43 | 低于同seed T2/K1；否定最终候选 |
| E025 / T2-S2 | T2额外seed 3333 | 63.31 | 64.95 | T2三seed基线组成 |
| E026 / K1-S2 | K1额外seed 3333 | 63.87 | 66.75 | 同seed优于T2；锁定K1最终结构 |
| E027 / T11 | K1 + FACR前置独立masked aggregation | 66.86 | 69.86 | seed 1111单次大幅提升；E028未复现 |
| E028 / T11-S1 | T11额外seed 2222 | 64.76 | 67.46 | 较同seed E023/K1主指标-0.17/-0.36；停止追加seed |
| E029 / T12 | K1 + 训练期共享跨模态token重建 | 65.82 | 67.58 | 相对同seed K1 +2.51/+0.59；进入seed 2222配对验证 |
| E030 / T12-S1 | T12额外seed 2222 | 66.94 | 68.90 | 相对同seed K1 +2.01/+1.08；不再追加RGBNT201 seed |

所有表中结果均为RGBNT201、batch 40、20 epoch、主结果关闭re-ranking；E001–E021、E027及E029使用seed 1111，E022–E024、E028及E030使用seed 2222，E025–E026使用seed 3333。融合实验每个epoch验证，以便与已有最佳epoch选择协议一致。

## 六、当前运行状态

- E030/T12-S1已正常完成：commit `0daf4d2`，returncode 0，耗时758.0秒；最佳epoch 16，66.94 mAP、68.90 Rank-1、79.43 Rank-5、85.77 Rank-10。相对同seed E023/K1提升2.01/1.08/0.84/1.80，两个主指标均超过预设对照；结果JSON、DONE、日志、配置快照、TensorBoard事件和最佳checkpoint均已保留，训练进程已退出。基于时间成本，不再运行T12 seed 3333。
- E029/T12已正常完成：commit `0daf4d2`，returncode 0，耗时757.7秒；最佳epoch 20，65.82 mAP、67.58 Rank-1、79.43 Rank-5、85.65 Rank-10。相对同seed E015/K1提升2.51/0.59/1.20/2.16，通过预设seed 2222晋级门槛；结果JSON、DONE、日志、配置快照、TensorBoard事件和最佳checkpoint均已保留，runner未生成预期的`retention.json`。
- E028/T11-S1已正常完成：commit `a78c007`，returncode 0，耗时763.2秒；最佳epoch 17，64.76 mAP、67.46 Rank-1、81.46 Rank-5、86.36 Rank-10；实际保留11.6706%。较同seed E023/K1的mAP和Rank-1分别低0.17和0.36，未通过预设门槛，不运行seed 3333。
- E027/T11已正常完成：commit `a78c007`，returncode 0，耗时771.7秒；最佳epoch 20，66.86 mAP、69.86 Rank-1、82.54 Rank-5、88.16 Rank-10；实际保留11.3782%。相对E015/K1同seed提高3.55/2.87/4.31/4.67，通过预设晋级门槛。

- E006–E008批量实验已完成；E008最佳epoch 14，mAP 62.70%、Rank-1 61.00%、Rank-5 75.00%、Rank-10 81.46%。
- E009/T3R已完成：807.5秒，最佳epoch 11，mAP 62.70%、Rank-1 64.23%、Rank-5 75.00%、Rank-10 80.50%。
- E010/T4已完成：809.4秒，最佳epoch 11，mAP 59.90%、Rank-1 59.81%、Rank-5 71.53%、Rank-10 78.59%，代码commit `1d0a520`。
- E010输出目录：`/root/autodl-tmp/outputs/HTL-ReID/E010_T4_facss_masked_facr_seed1111`；已生成`DONE`、结果JSON和最佳checkpoint。
- E011/T5已完成：772.3秒，最佳epoch 18，mAP 62.37%、Rank-1 63.04%、Rank-5 76.08%、Rank-10 83.37%；平均0.2842秒/batch，相对T2仅增加约1.5%。
- E012/T6已完成：792.8秒，最佳epoch 13，mAP 60.04%、Rank-1 60.17%、Rank-5 72.85%、Rank-10 80.50%；最佳checkpoint的argmax K=4，最终并集保留27.10%。
- E013固定K代理筛选已完成：K=1/2/4/8/16的mAP依次为59.6751/59.8778/60.0353/60.0675/60.0716；结果差异很小，仅用于选择E014候选。
- E014/T7固定K=16已完成：770.2秒，最佳epoch 16，mAP 63.80%、Rank-1 65.07%、Rank-5 75.96%、Rank-10 82.06%；输出目录`/root/autodl-tmp/outputs/HTL-ReID/E014_T7_sfts_fixed_k16_facr_seed1111`。
- E015/K1已完成：818.1秒，最佳epoch 17，mAP 63.31%、Rank-1 66.99%，实际保留11.53%。
- E016/K2已完成：801.5秒，最佳epoch 18，mAP 62.35%、Rank-1 64.59%，实际保留17.38%。
- E017/K4已完成：788.6秒，最佳epoch 17，mAP 62.83%、Rank-1 64.71%，实际保留26.71%。
- E018/K8已完成：769.5秒，最佳epoch 17，mAP 62.18%、Rank-1 63.76%，实际保留41.64%。
- E019/T8已完成：769.7秒，最佳epoch 16，mAP 62.68%、Rank-1 64.23%、Rank-5 76.32%、Rank-10 82.66%；实际保留60.37%，路由无明显塌缩，否定路由均衡损失。
- E020/T9已完成：733.8秒，最佳epoch 14，mAP 62.33%、Rank-1 61.12%；否定全tokens FACR后的无条件自模态细化。
- E021/T10已完成：764.6秒，最佳epoch 17，mAP 63.88%、Rank-1 66.27%、Rank-5 77.03%、Rank-10 83.25%；实际保留11.44%，但额外seed未复现。
- 共享ViT三模态批处理优化在服务器正式尺寸基准中为0.2680秒/step，慢于顺序版0.2573秒/step约4.2%，已通过commit `c8e7a71`回滚；fused AdamW相对foreach仅约0.4%差异，不采用。
- seed 2222配对任务E022–E024均已正常完成，returncode均为0。
- E022/T2：721.5秒，最佳epoch 17，64.41 mAP、66.15 Rank-1、79.78 Rank-5、85.65 Rank-10。
- E023/K1：743.5秒，最佳epoch 10，64.93 mAP、67.82 Rank-1、78.59 Rank-5、83.97 Rank-10；实际保留11.6336%。
- E024/T10：761.3秒，最佳epoch 17，63.62 mAP、65.43 Rank-1、75.96 Rank-5、81.82 Rank-10；实际保留11.6685%。
- 批次日志：`/root/autodl-tmp/outputs/HTL-ReID/paired_seed2222_E022-E024.runner.log`；各实验的结果JSON、日志和最佳checkpoint均已保留，E023/E024的`retention.json`已补齐。
- seed 3333第三组统计任务E025/E026已于2026-08-26 23:10 CST全部完成。
- E025/T2已正常完成：722.3秒，最佳epoch 18，63.31 mAP、64.95 Rank-1、78.59 Rank-5、85.41 Rank-10。
- E026/K1已正常完成：754.8秒，最佳epoch 15，63.87 mAP、66.75 Rank-1、79.31 Rank-5、85.41 Rank-10；实际保留11.5181%。
- E025/E026 returncode均为0，原批次编排进程已退出。批次日志：`/root/autodl-tmp/outputs/HTL-ReID/paired_seed3333_E025-E026.runner.log`；结果JSON、最佳checkpoint及E026 `retention.json`均已保留。
- 最终三seed统计：T2为63.847±0.550 mAP、65.390±0.661 Rank-1、78.627±1.135 Rank-5、85.010±0.909 Rank-10；K1为64.037±0.823、67.187±0.561、78.710±0.550、84.290±0.999。
- T12前两个seed统计：66.380±0.792 mAP、68.240±0.933 Rank-1、79.430±0.000 Rank-5、85.710±0.085 Rank-10（样本标准差）；相对配对K1的两次mAP增益为+2.51/+2.01，Rank-1增益为+0.59/+1.08，方向一致。

## 七、下一步

1. T12作为当前优先结构，K1保持为冻结基线和同协议参照，不再改变T11结构；
2. E028未复现T11对K1的主指标优势，按预设规则不运行T11 seed 3333；T11仅作为单seed正向但不稳定的消融；
3. E030已通过配对验证；停止RGBNT201上的T12额外seed与重建权重搜索；
4. 后续优先使用冻结T12配置进入RGBNT100和MSVR310跨数据集验证，不再以同数据集重复seed消耗训练时间；
5. K1保留为稳定参照；如论文必须补充第三seed，只在用户明确授权后讨论，不默认执行；
6. 完成M0与T12/K1推理路径的参数量、GFLOPs、延迟和显存对比；K1 mask未物理压缩FACR输入，不声称等比例效率收益；
7. 用三seed统计更新论文方法、消融表、摘要和回复信，主结果报告mean±std。

可选性能上限方向（未实现、未授权运行）：当前T12对全部288个patch等权计算重建损失，背景区域可能稀释身份监督。如果跨数据集前仍明确要求只做一次RGBNT201改进，唯一优先候选是使用`detach`后的SFTS共享mask对逐token余弦重建损失加权：被选token高权重、未选token保留低权重并按权重和归一化；总损失系数继续固定0.1，不做网格搜索、不增加推理计算、只按单seed筛选。默认计划仍是直接转跨数据集，不自动启动该实验。

## 八、统一约束

- 每个训练进程必须使用`timeout --signal=TERM --kill-after=10s 30m ...`；
- RGBNT201 T2/K1三seed已使用1111、2222、3333完成，固定20 epoch、batch 40及相同预训练与优化协议；不得追加seed追逐结果；
- 时间优先：新结构默认单seed筛选，只有结果明确且确有决策价值时才做一次配对复验；除非用户明确要求，不自动追加第三seed，优先把预算用于跨数据集验证和论文必需实验；
- 主结果明确关闭re-ranking；
- 正式或失败运行均登记独立E编号并保留配置、日志、结果JSON和最佳checkpoint；
- 不删除远端训练产物，清理前必须先列出候选和大小并取得用户同意；
- 不把引用模块通过小改名包装成原创，TPM必须明确引用TOP-ReID。

## 九、新会话读取顺序

1. 本文档；
2. `/Users/a123/Documents/reid/HTL-ReID/实验记录.md`；
3. `/Users/a123/Documents/reid/HTL-ReID/实验记录/E025_T2_RGBNT201_seed3333.md`、`E026_K1_RGBNT201_seed3333.md`及`实验记录.md`三seed汇总，再按需读取E007、E015、E022和E023；
4. 如需代码细节，再读`HTL-ReID/modeling/fusion_part/SFTS.py`、`TPM.py`和`modeling/make_model.py`；
5. 确认E029/E030训练代码快照为`0daf4d2`；K1实现保持兼容，T11/T12能力均默认关闭并由独立配置开启；E027–E030均已完成，不得重复启动；
6. T11稳定性验证已停止；T12不再运行RGBNT201 seed 3333，下一步优先转跨数据集验证。
