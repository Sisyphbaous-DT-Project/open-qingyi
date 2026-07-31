# QINGYI-KDA-0.6B 技术报告

**从全注意力到 KDA：0.6B 混合线性化的手术、蒸馏与人格对齐全记录**

DT-Project · QOSP（Qingyi Open Source Project，开源清漪计划）
2026-07-31 · 代码：https://github.com/Sisyphbaous-DT-Project/open-qingyi · 权重：https://huggingface.co/shiershuihesaixiliya/qingyi-kda-0.6b

---

## 摘要

我们报告一条将 Qwen3-0.6B-Base 的 28 层全注意力中的 21 层替换为 KDA（Kimi Delta Attention）线性注意力、保留 7 层原生 GQA 的完整转换管线，以及随后恢复能力、注入人格的全部实验。项目历经三代手术（v1/v2/v3）、五次训练范式迭代，最终模型 C-Eval 从 Stage 3a 蒸馏后的 28.83% 恢复至 41.83% ± 1.33（教师基线 50.59%），全程未使用 C-Eval 等基准题背题，也未依赖传统 CE-CPT 大规模重建知识。

三个核心发现：(1) **接口伤是低分的主要可修复因素之一**——四选项置换诊断证明转换后的模型严重"黏住 A/B/C/D 标签"而非跟随答案内容，1,000 步格式专项训练即可恢复 12.5 个百分点；这是接口失能的强证据，但不排除并存的知识损伤；(2) **人格可以"白嫖"**——15 个 C-Eval 守护点跟踪确认人格 SFT/小灶/DPO 全程未造成灾难性遗忘（SFT 首点短暂下探 2.7pt 后恢复）；(3) **on-policy DPO 一轮即甜区**——第二轮迭代只增生复读而无收益。

本报告同时完整公开三代手术的失败与诊断过程：v1 的伪 teacher-init、v2 的"自信失忆者剪刀差"、v3 的门控消融（g1→g6），以及 bf16 优化精度吞更新、KDA 增量缓存事故等工程教训。

## 1. 引言与目标

线性注意力（linear attention）以 O(1) 循环状态解码替换全注意力的 O(n) KV 缓存，在长上下文与边缘部署上有结构性优势。KDA（Kimi Delta Attention）是 Moonshot 在 Kimi Linear / K3 系列中使用的 delta-rule 线性注意力变体，引入 per-channel 遗忘门控与 beta 写入门。

已有工作中，注意力→线性注意力的"线性化转换"（linearization）并不新鲜。**与本项目同方向的公开成功工作已经存在**：GenDistill（arXiv:2603.26556）明确完成了 Qwen3-0.6B→Hybrid-KDA 转换，其分层对齐 + 端到端 KL（Stage 3a）+ completion-only KD（Stage 3b）配方与本项目管线同族，本项目的解冻范围设计即直接采用其 MLP 冻结消融结论。另一方面，公开记录中也有失败案例：THUNLP 的 HALO 管线报告 Qwen3→KDA 在其 Appendix B 同配置实验中 Stage 2 梯度范数发散（gnorm→inf），降低学习率无效（HALO App. G.2）。在此坐标下，本项目的贡献是：(1) 消费级/租用单卡预算内的独立工程实现与全流程事故/诊断记录；(2) GenDistill 未覆盖的"接口伤"四置换诊断学；(3) 转换后人格对齐（SFT/小灶/DPO）的扩展实验。

本项目的目标有三：

1. 在消费级/租用单卡预算内，端到端跑通 Qwen3-0.6B→KDA 混合转换并收敛；
2. 定量回答"转换到底伤了什么"——知识、接口还是推理链；
3. 验证人格能否作为权重级属性在转换后注入并保持（数字孪生场景）。

基座选择 Qwen3-0.6B-Base：Apache 2.0、教师基线 C-Eval 50.59% ± 1.33（本报告全部 C-Eval 均为 lm-eval `ceval-valid`、0-shot、loglikelihood 选项排序口径），小到可以在 32GB 单卡上做全管线消融。

## 2. 方法总览

### 2.1 混合布局

28 层中每 4 层保留 1 层原生 GQA（第 3, 7, 11, 15, 19, 23, 27 层，0-indexed，末层必为全注意力），其余 21 层替换为 KDA。KDA 层逐行复刻 Kimi-Linear 官方实现：16 头、causal short conv（kernel 4）+ SiLU、q/k L2 norm、低秩 forget gate、beta sigmoid 写入门、低秩 sigmoid 输出门 + FusedRMSNormGated、NoPE。fla `chunk_kda` 与参考实现对拍数值误差 ~1e-6。注意整模型仍为混合架构：KV cache 随保留的 7 层 GQA 继续线性增长，O(1) 固定循环状态只覆盖约 75% 的层。

### 2.2 训练管线（最终版）

```
Qwen3-0.6B-Base
 → 手术替换（v3，gates g6）
 → Stage 2  分层对齐：hidden-state MSE，教师冻结，误差层间 detach
 → Stage 3a 端到端 forward-KL 蒸馏（T=1，alpha_CE=0）
 → Stage 3b 格式专项：completion-only KL（选择题四置换/古诗/翻译）
 → SFT      人格语料两 epoch
 → 身份小灶  身份 QA 高浓度短训
 → DPO      on-policy 偏好对抛光
```

### 2.3 评测体系

- **三档尺**（tune/valid/release）：文档级指纹隔离，与训练窗口零重合，调参只用 tune，报告只用 valid，release 档全项目锁定隔离、最多只看一次；
- **C-Eval 守护点**：人格训练每 500 步一次 C-Eval，红线为 SFT 前水平的噪声带（±1.33）；
- **生成探针**：固定 13~20 题贪心 + 采样双轨，覆盖身份绑定、拒密、通用事实、闲聊；
- **四置换诊断**：同一道选择题循环移动四个选项位置四次，区分"跟随答案内容"与"黏住选项标签"。

## 3. v1：首次端到端跑通（2026-07-26 ~ 07-29）

### 3.1 手术与蒸馏

v1 手术按上述布局替换，KDA 配置 head_dim 64。原记录声称以教师 Q/K/V/O 投影初始化（teacher-init），**2026-07-29 法医验证推翻**：distill-best 的 k_proj 与教师投影余弦 −0.0000、与随机初始化余弦 +0.6948——投影迁移从未实际发生，v1 实为随机初始化。

蒸馏（distill）：逐层 hidden-state MSE 对齐，教师冻结、只训 21 个 KDA 层，49,152 tok/step（micro 4 × seq 2048 × ga 6），lr 1e-3→1e-4 cosine。本地 RTX 4070 Laptop 8GB，~2,640 tok/s。跑至 step-2875，best = step-2750，此时 CE gap +5.08（轨迹 7.98→5.08，MSE 1.35→0.62），step-2000 后进入平台期。

CPT：从蒸馏 best 冷启动，全模型解冻，49,152 tok/step，lr 3e-4→3e-5，wd 0.1，数据为英文 fineweb-edu-dedup 60% + 中文 IndustryCorpus2 40%。**step-750 即收工**：held-out CE gap 从 +7.49 收敛到 **+0.1983**（student 2.8393 vs teacher 2.6410），本地约 9 小时，预算仅为 HALO 管线的 ~6.5%。

### 3.2 v1 成绩单与代价

| 模型 | C-Eval | MMLU |
|---|---|---|
| Qwen3-0.6B-Base | 50.52% ± 1.33 | 50.40% ± 0.41 |
| v1 CPT | 22.81% ± 1.15 | 23.39% ± 0.36 |
| v1 人格版 | 21.99% ± 1.13 | — |

CE 口径上 CPT 已逼近教师（gap +0.1983），但选择题跌穿随机线（~25%）。这一"CE 正常、选择题死亡"的剪刀差是 v2/v3 时代全部诊断工作的起点。

### 3.3 v1 人格线

- **SFT 主训 11,631 步**（3 epoch，31,212 条：个性化 47.7% + smoltalk2 + coig），bf16 + 8bit AdamW + liger fused CE，云 4080S 32GB，epoch ≈2.4h。held-out CE 4.78→3.8390。结果：语气人格烘焙成功，身份事实绑定失败。人格税 ≈0（21.99% vs 22.81%，误差线内）；
- **身份小灶** 8,440 条（222 独特 QA ×20 + 4,000 聊天），1,030 步，step-600 绑定成功（自名清漪、作者号码绑定、api key 拒绝）；
- **DPO** 1,252 对 on-policy，286 步 16 分钟，held-out reward acc 0.560→0.800；
- **DPO 第二轮（模型名小灶）失败**——216 对强灌模型名，结果模型名仍念错、"生气"口癖侵入闲聊，回滚。教训：DPO 是偏好微调不是记忆工具；
- **name-fix v1 失败 → v2 配平成功**：目标串占比 49% 漫溢到号码/api key 题；降至 26% 并加保护性 QA 后四项全对（见 §8.2 对"浓度阈值"经验的修正）；
- **Canary 提取测试 PASS**：347 个真实姓名零自发泄漏、逐字提取 0/15。

### 3.4 v1 工程事故（教训入选）

- **KDA 增量缓存 bug（最大隐蔽事故）**：`KDALayer.forward` 接收 cache 但直接忽略，带缓存生成时第二 token 起坍缩成乱码——此前多轮"人格未绑定"的判读部分是此 bug 假象。修复为 `HybridKDACache`（每层循环状态 S + 3 个 conv 窗口）：乱码坍缩消除、加速 1.7×、无需重训。如实记录：cached vs full 的五题贪心对拍一致率为 100/25/70/100/100（验证对象为旧 KDA64 权重；非满分项为数值路径级差异而非行为坍缩，最终 KDA128 权重未重做该项对拍）；
- **Triton 版本**：云镜像自带 triton 3.7.1 使 `chunk_kda` 编译后训练假死，锁定 triton==3.6.0；
- **训练中并发评测**：显存顶穿、吞吐不可逆下降，立规"先停训练再测"；
- **`nohup python` 无 `-u`**：stdout 块缓冲造成卡死假象，白调试约 1 小时，立规全部 `python -u`；
- **SFT 性能**：micro_batch=1 时 CPU dispatch 占 90%，改 micro_batch=8 + 长度分桶后 epoch 6.9h→2.4h。教训：先 profile 再优化。

## 4. v2：真 teacher-init 与 CPT+KL 联合（2026-07-29 晚）

v2 手术与 v1 唯一差异是初始化：真正执行教师 Q/K/V 投影移植。蒸馏同管线跑至 step-750（MSE 0.685），与 v1 的 MSE 曲线全程重合；C-Eval 25.33%→24.29% 随机线躺平。**结论一：MSE 蒸馏阶段不转移知识，与初始化无关。**

随后启动 CPT+KL 联合训练（3000 步计划，lr 1e-4，λ 1.0，数据 zhwiki 0.50 / fineweb-edu 0.35 / IndustryCorpus2 0.15）：起点学生 CE 9.18（gap +6.17）、KL 6.32，封盘于 step-1275（CE 3.5658，gap +0.5579，KL 0.7063）。但 C-Eval 全程 23.1~23.9 随机线下方筑底。

**核心发现："自信失忆者剪刀差"**——KL 持续下降（分布整体逼近教师），但选择题上模型越自信越错。这证明分布级的 KL 收敛不等于知识可用性恢复，单看 CE/KL 会给出虚假信心。v2 当晚封盘暂停。

**事后（v3 时代）回溯诊断**证伪 v2 的两个根基：

1. v2 的"真 teacher-init"同样是错接——v2-init 与教师输出功能探针 mean cosine 仅 0.0069，**与教师输出正交**；
2. v2 手术存在三大结构缺陷：KDA 层缩容（4.49M vs 原生 6.29M）、头数错接、以及致命的强遗忘门控——默认 dt_bias −4、exp(A_log)∈(0.001, 0.01) 使遗忘半衰期极短，新尺 overall CE 高达 16.60（均匀随机为 11.93，教师为 3.01）。

## 5. v3：诊断驱动重构（2026-07-30）

v3 不再改训练，先修手术本身。

### 5.1 门控消融（g1→g7）

以新尺 overall CE 为判据（v2=16.60，随机=11.93，教师=3.01）做 forget-gate 初始化网格：

| 方案 | dt_bias | exp(A_log) 区间（衰减尺度） | CE |
|---|---|---|---|
| g1（v2 默认） | −4 | (0.001, 0.01) | 14.556 |
| g5 | −2 | (0.01, 0.1) | 11.901 |
| g3 | +2 | (0.1, 1) | 10.179 |
| g7 | 0 | (0.005, 0.05) | 10.490 |
| g2 | 0 | (0.01, 0.1) | 9.740 |
| **g6（采用）** | 0 | **(0.03, 0.3)** | **9.4756** |

规律清晰：近全保留→循环状态爆炸；近全遗忘（v2 式）→比随机还差。g6 实测半衰期约 3.3~31.6 token（中位 6.3）。v3 最终功能探针 mean cosine 0.4343（v2-init 的 63 倍），relL2 ≈1.86。

### 5.2 out_scale 否决实验

给输出门 `o_norm.weight ×2` 补偿幅度砍半：CE 反而 +1.41，否决，保持 out_scale=1——尺度问题留给训练学，不靠手术硬补。

### 5.3 流程硬化

- **canonical 锁定**：批准版初始权重固定 SHA-256，训练入口 `--expect-init-hash` 硬校验，杜绝"云端重新随机构建的差不多版"；
- **FP32 master**：实测 bf16 参数 + bf16 动量下，初值 1.0 的 norm 权重在 lr 1e-4 跑 1000 步仍精确等于 1.0（半 ULP 吞更新）。Stage 2/3 全部改 FP32 master 参数与动量；
- **三档尺隔离**（见 §2.3）；
- **可恢复性**：数据游标逐 token 一致恢复 + resume 11 项超参校验（lr/warmup/weight_decay/kl_chunk 等），防静默续错。

### 5.4 Stage 2（分层对齐，正式版）

LayerAligner：21 层 KDA 各拿对应教师 hidden，教师冻结、误差层间 detach、仅训 KDA（193.6M）。lr 1e-4→1e-5，micro 2 × accum 2 = 8192 tok/step。核心曲线：手术初值 overall CE 9.4756 → best step-900 valid CE 4.1250、KL 1.33——**但此时 C-Eval 仍只有 25.48%（随机线）**。这是本报告"perplexity 会撒谎"主题最干净的单个证据：层对齐把 hidden 轨迹大幅拉近教师，选择题能力却纹丝不动。Stage 2 是"待康复起点"而非终点，功能继承由 Stage 3 完成。

## 6. Stage 3a：端到端 KL 蒸馏（2026-07-31）

- 目标：纯 forward KL（teacher‖student），T=1，alpha_CE=0，最终 hidden 过 lm_head 计算，梯度沿完整前向图回传全部 21 层 KDA；
- 解冻范围 457.9M（KDA + 全部 MLP + block/final LayerNorm；embedding/tied lm_head/7 层保留 attention 冻结）——GenDistill 消融表明冻结 MLP 会显著损失知识通路（C-Eval 37.9→31.4）；
- 工程：绕过 CausalLM 完整 logits（单份 bf16 logits 2.32GiB），直接取 `last_hidden_state`；ChunkedKL 分块 fp32 计算，与 dense 对照梯度误差 ≤2.4e-7；
- 配置：lr 2.5e-5 余弦至 2.5e-6，8192 tok/step，micro 2 × accum 2（micro 4 在 optimizer 状态分配后评测 OOM，实测峰值 20.4/21.3 GiB）；
- **主臂 7,000 步 / 57.344M tokens**。step-7000：valid CE 2.9385（教师 2.8108，gap +0.1277），valid KL 0.1602；C-Eval 28.83%；
- **温度对照（A/B 臂从同一 step-7000 检查点、同数据游标、同优化器状态各续训 1,000 步）**：A/T=1 终值 C-Eval 28.68%，B/T=2 终值 30.46%，T=2 连续两个检查点站上 30%——方向性地改善选择题选项排序；代价是 valid CE/KL 从 2.9385/0.1602 恶化到 2.9485/0.1702（通用分布拟合受损）。约 1.8pt 差距未做显著性检验，只能作为方向性观察，不作强因果结论。

step-7000 的 C-Eval 28.83% 看似原地踏步，但 §7 的诊断证明这是假象。

## 7. 核心实验：接口伤诊断与 Stage 3b 修复

### 7.1 四置换诊断

从训练窗口原文（中文维基 + IndustryCorpus2）生成 161 道干净选择题（文档级隔离、与 C-Eval/MMLU/CMMLU 精确及 5-gram 零重合、与三档尺零重合），每题循环移动四个选项位置四次：

| 指标 | 教师 | 学生（step-7000） |
|---|---|---|
| accuracy（161×4） | 66.46% | 36.18% |
| 正确项平均 margin | +0.9953 | **−0.1167** |
| 预测选 A 占比 | — | **81.06%** |
| 四次换位仍黏同一标签 | — | **106/161** |
| 恰好只答对 1/4（碰巧） | — | 112/161 |

面对"答案："时学生压倒性输出 A，选项换位后不跟随答案内容。普通预训练文本上的 T=1 KL 几乎不提供多选格式监督。需要强调因果边界：四置换直接证明的是**严重的标签黏着（接口失能）**，它并不能直接证明学生原本"认识答案内容"——原始诊断日志也注明缺少不经过 A/B/C/D 标签的答案文本直接评分。因此严谨的表述是：**接口伤是低分的主要可修复因素之一**（§7.2 的阶跃提供强证据），但不排除并存的知识损伤。

### 7.2 Stage 3b 格式专项

数据：6,250 道教师验证 MCQ（教师四次换位全对且 min margin ≥0.25）×4 置换（A/B/C/D 严格各 25%）+ 3,000 古诗问答 + 3,000 翻译；与全部基准及三档尺零重合验证；completion-only KL（prompt 位置 mask=0，不跨 pack）。

从 step-7000 起跑 1,000 步（T=2），结果：

- **C-Eval 28.83% → 41.31%**（+12.48pt，四个检查点持续站 39~41%）；
- 四置换复测：accuracy 36.18% → **53.73%**，四次换位全对 7/161 → **47/161**，正确项 margin −0.1167 → **+0.2330**，选 A 率 81.06% → 58.5%，黏标签题 106/161 → 47/161——**显著修复，但未治愈**（A 率仍远高于 25% 的均衡值）；
- 代价：valid CE 2.9385 → 2.9763，valid KL 0.1602 → 0.1959——+12.48pt 不是免费的，通用分布拟合小幅让位；
- 结论：C-Eval 所需的大部分知识并未被不可逆抹除，主要瓶颈是知识到选项标签的接口；这不等价于"知识零损失"（见 §10.2）。

## 8. 人格训练与对齐（v3 时代）

### 8.1 SFT 两 epoch（7,768 步）

数据 31,068 条（真实聊天清洗 + 合成问答），held-out 200 条。lr 第一 epoch 1.5e-5，第二 epoch resume 时显式改为 1e-5；micro 8，12.3M tok/epoch。held-out CE 2.9104→2.5298 单调收敛。**15 个 C-Eval 守护点：首个点（step-500）跌至 38.63%（−2.68pt，跌出噪声带），随后恢复并稳定在 39%~41%**，终值 40.34%（SFT 前 41.31%，−0.97pt 在噪声带内），峰值 40.86%。

### 8.2 身份小灶（300 步）

数据 8,975 条 = 249 独特身份 QA ×20 + 4,000 聊天。逐 100 步验收：

- step-100：元素激活（"清漪"首现自称位但语义乱）；
- **step-300：点燃**——自名清漪、模型名 QINGYI-KDA-0.6B、否认 ChatGPT 三要素全绑定，held-out CE 2.4647，通用零退化；
- step-400：漫溢早期形态——事实题开始否认 H₂O、闲聊贪心循环塌陷、system prompt 开始编造内容。按预注册规则停训，弃 400 保 300。

**浓度安全窗口 ~300 步**（本轮条件：identity 占比 55% + 4,000 条保护性聊天样本）。v1 name-fix 时代"目标串占比 >20% 即漫溢"的经验在本轮被修正：v1 曾在 26% 配平成功、本轮 55% 在 300 步内成功——真正的变量是**浓度 × 训练步数 × 保护样本的组合**，不存在普适的单一浓度阈值。

### 8.3 DPO 抛光（100 步）

on-policy 数据：当前模型对 248 道坑题 T0.9 采样 ×4 + 300 正常聊天 ×2，抓真实失败为 rejected（如密码题交出号码串、api key 题崩日语乱码），组 545 pairs。lr 1e-6，β 0.1，ref 冻结：

- held-out reward acc 0.625→**0.792**（step-100 best），margin 0.07→0.29；
- **第二轮迭代（再锚定 best）无效**：acc 0.646 未回 0.792，生成复读增生，整轮废弃。一轮即甜区；
- 贪心探针身份三要素完好；拒密题（api key）贪心仍未翻转——数据级问题，DPO 磨不平。

### 8.4 最终验收

| 项 | 值 |
|---|---|
| **C-Eval（最终权重）** | **41.83% ± 1.33** |
| 教师基线（同口径） | 50.59% |
| 参数量 | 657,506,000（约 0.658B；"0.6B" 为系列名） |
| 采样稳定性 | T0.7 下身份绑定约半数回潮；更低温度未实测，T≤0.5 仅为部署建议（外推，非实测结论） |
| 权重 SHA-256 | f375a99f…78e51a |

## 9. 结果总表

| 阶段 | 检查点 | C-Eval | 备注 |
|---|---|---|---|
| 基座 | Qwen3-0.6B-Base | 50.52% ± 1.33 | 教师 |
| v1 | CPT step-750 | 22.81% ± 1.15 | CE gap +0.1983 但选择题死亡 |
| v1 | 人格版 | 21.99% ± 1.13 | 人格税 ≈0 |
| v2 | CPT+KL step-1275 | ~23.1% | 自信失忆者剪刀差 |
| v3 | 手术初值（未训练） | — | 新尺 overall CE 9.4756 |
| v3 | Stage 2 best step-900 | 25.48% | CE 4.1250 / KL 1.33，选择题仍随机线——"perplexity 会撒谎"最干净证据 |
| v3 | Stage 3a step-7000 | 28.83% | valid CE gap +0.1277 |
| v3 | Stage 3b step-8000 | **41.31%** | 接口修复，+12.48pt |
| v3 | SFT step-7768 | 40.34% | 守护点首点 38.63% 后恢复 39~41% |
| v3 | 小灶 step-300 | **42.27%** | 身份点燃，全程最高 |
| v3 | DPO step-100（最终） | **41.83% ± 1.33** | Δ vs 小灶 −0.44pt（噪声带内） |

## 10. 讨论

### 10.1 与相关工作的关系

- **vs GenDistill（arXiv:2603.26556）**：同方向的已公开成功工作，已完成 Qwen3-0.6B→Hybrid-KDA（Stage 3a/3b、completion-only KD）。本项目是其配方族在消费级预算下的独立工程实现（解冻范围直接采用其 MLP 冻结消融结论），贡献在于其未覆盖的接口伤四置换诊断学、人格对齐扩展，以及全流程事故/修复记录；
- **vs HALO**：HALO 的公开记录中，Qwen3→KDA 按 Appendix B 同配置实验在 Stage 2 梯度发散失败；本管线端到端收敛。需要纠正一个容易写错的对照：HALO 正式管线同样保留 25% 注意力层，因此"混合保留 vs 全替换"不构成两家差异。收敛差异的**候选解释**（未经消融证明）包括：门控初始化（g6）、FP32 master（bf16 下 norm 权重冻结不更新，与"降 lr 无效"的症状一致）、梯度裁剪与 canonical hash 锁定流程——值得 HALO 作者验证；
- **转换预算**：v1 CPT 仅 HALO 管线的 ~6.5%；v3 主轴线（Stage 2 + 3a + 3b）约 74M tokens（57.3M + 8.2M + Stage 2 约 8M），同样远低于同类工作的 500M~2.3B 预算——0.6B 规模 + 混合保留布局可能是低预算收敛的两个来源。

### 10.2 局限

- 最终与教师仍有 ~8.8pt C-Eval 差距；知识类问答（古诗、翻译）存在教师对而学生错的残例，接口修复不等于知识零损失；
- "数据多样性不足是能力上限的主因、而非架构上限"目前只是**工作假设**：单模型、单种子、单数据规模无法排除 KDA 容量或架构上限；
- **评测反馈与 winner's curse**：同一 `ceval-valid` 被反复用于检查点选择（守护点 15+ 次），最终 41.83% 不是 untouched test，存在选择偏差；
- **release 尺未开封**：三档中的 release 档已锁定隔离，但截至发稿未执行最终归档评测，本报告全部数字均出自 valid 档；
- 最终权重只完整复测了 C-Eval：缺最终 MMLU/CMMLU、标准生成基准、长上下文、吞吐与 KV 显存实测；最终人格权重也未重跑 canary 提取测试（v1 时代 PASS 的结论不能直接继承到 v3）；
- 混合布局仍保留 7 层 GQA：整模型 KV cache 仍随上下文线性增长，O(1) 固定循环状态只覆盖约 75% 的层；
- 拒密行为在采样下不稳定（会编造假 key 应对，虽无真实密钥可泄）；
- KDA 层无 CPU 回退（Triton kernel），端侧部署需 GPU；
- 0.6B 规模结论外推到更大模型需重新验证。

### 10.3 工程经验（可迁移）

1. **bf16 训练小更新会被吞**：norm 类权重（≈1.0）在 bf16 参数+动量下可能数百步不动——任何蒸馏/微调项目若"gnorm 正常但权重不变"，先查 FP32 master；
2. **CE/KL 会撒谎**：分布收敛与下游能力可长期背离（v2 剪刀差、Stage 2 的 25.48%），必须配任务级守护评测；
3. **评测尺要防自污染**：三档隔离 + 指纹去重 + 选项置换，缺一个都可能高估；
4. **no cache 的架构改造**：新层必须第一天就实现增量缓存并对拍 full/cached，否则生成期 bug 会污染全部行为判读。

## 11. 结论

我们完整记录了一个 0.6B 全注意力模型转换为 KDA 混合架构并恢复能力、注入人格的全过程：三代手术、两次范式推翻（MSE 无效论、KL 收敛≠能力）、一次诊断驱动的修复（接口伤四置换 → +12.48pt）。最终模型以 41.83% C-Eval、权重级人格收官，代码与权重以 Apache 2.0 开源（隐私聊天数据不公开；第三方数据集许可以各自源为准）。全部配方、模型卡与本报告同步发布。

## 12. 参考文献

1. Qwen Team. Qwen3 Technical Report. arXiv:2505.09388；基座权重：https://huggingface.co/Qwen/Qwen3-0.6B-Base
2. Moonshot AI. Kimi Linear / KDA（Kimi Delta Attention）技术报告与 K3 Tech Report（KDA 参考实现来源）.
3. fla: Flash Linear Attention 库（`chunk_kda` 等 Triton kernel）. https://github.com/fla-org/flash-linear-attention
4. GenDistill: 注意力→线性注意力分层对齐 + 端到端 KL 蒸馏. arXiv:2603.26556.
5. THUNLP. HALO linearization pipeline（本文引用其 Appendix B 配置与 Appendix G.2 失败记录）.
6. Huang et al. C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models. arXiv:2305.08322.
7. Hendrycks et al. Measuring Massive Multitask Understanding (MMLU). arXiv:2009.03300.
8. EleutherAI. lm-evaluation-harness. https://github.com/EleutherAI/lm-evaluation-harness
9. 训练数据：中文维基、fineweb-edu-dedup、IndustryCorpus2（酒店/汽车域）、SmolTalk2、COIG-CQIA（许可情况见各自数据源）.

## 附录 A：硬件与预算

| 阶段 | 硬件 | 规模 |
|---|---|---|
| v1 蒸馏 | RTX 4070 Laptop 8GB | 2,875 步 × 49,152 tok（~141M tok），~48h 墙钟 |
| v1 CPT | 同上 | 750 步，~9h |
| v1 人格线 | 云 4080S 32GB（¥1.77/h） | SFT 3 epoch ≈7h；DPO 16min |
| v2 CPT+KL | 云 4080S 32GB | 1,275 步 × 49,152 tok |
| v3 Stage 2 | 云 32GB vGPU | ≤1,000 步 × 8,192 tok（best = step-900） |
| v3 Stage 3a | 同上 | 7,000 步 × 8,192 tok ≈ 57.3M tok；A/B 温度对照臂各 +1,000 步 |
| v3 Stage 3b | 同上 | 1,000 步 ≈ 8.2M packed tok |
| v3 人格线 | 同上 | SFT 7,768 步；小灶 300 步；DPO 100 步 |

## 附录 B：数据

- 预训练/蒸馏：中文维基、fineweb-edu-dedup、IndustryCorpus2（酒店/汽车域）；
- Stage 3b：训练窗口原文生成 MCQ（教师四次换位全对且 margin≥0.25 筛选）+ 古诗/翻译问答；
- SFT/小灶：真实聊天记录清洗（不公开，隐私）+ 合成身份 QA（配方公开，见 scripts/）；
- DPO：on-policy 采样 + 标准答案；
- **污染审计口径说明**：强审计仅覆盖 Stage 3b 数据——对 26,902 道 C-Eval/MMLU/CMMLU 参考题精确及 5-gram 近重复零违规、与 4,200 个三档尺文档指纹零重合，§7.1 诊断题的 73 篇源文档也已被出题脚本排除。真实聊天、SmolTalk、COIG、DPO 偏好对等数据**未做同等级基准污染审计**（其中真实聊天与合成身份 QA 为自产数据，含基准题的可能性低，但我们不把"未审计"写成"已审计"）。

## 附录 C：时间线

| 日期 | 事件 |
|---|---|
| 07 上旬 | KDA 参考实现 + fla 对拍 + v1 手术 |
| 07-26~28 | v1 蒸馏 2,875 步（本地）→ CPT 750 步收工 |
| 07-28 晚 | v1 人格线：SFT 11,631 步 → 小灶 → DPO → name-fix 收官 |
| 07-29 | Canary PASS；三模型基准对照；v1 开源；晚 v2 手术 + CPT+KL 至 step-1275 封盘 |
| 07-30 | v3：回溯诊断（teacher-init 证伪、剪刀差归因）→ gates 消融定 g6 → 流程硬化 → Stage 2 |
| 07-31 | Stage 3a 7,000 步 → A/B 温度对照（各续至 step-8000）→ 接口伤四置换诊断 → Stage 3b 1,000 步（C-Eval 41.31%）→ SFT 7,768 步 → 小灶 300 步 → DPO 100 步 → QOSP 开源发布 |
