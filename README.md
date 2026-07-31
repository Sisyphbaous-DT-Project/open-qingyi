# open-qingyi — QOSP（Qingyi Open Source Project / 开源清漪计划）

> 把全注意力模型"做手术"转成 KDA 混合架构，再把能力和人格一点点训回来的完整配方。

**QINGYI-KDA-0.6B**：以 Qwen3-0.6B-Base 为基座，将 28 层全注意力中的 21 层替换为 KDA（Kimi Delta Attention）线性注意力层，保留 7 层原始 GQA，经分层对齐、端到端知识蒸馏、人格 SFT 与 DPO 抛光得到的混合架构模型。

## 血缘链

```
Qwen3-0.6B-Base
  → v3 手术（21×KDA + 7×GQA）
  → Stage 2 分层对齐（hidden state MSE）
  → Stage 3a 端到端 KL 蒸馏（8000 步 / 65M tokens）
  → Stage 3b 格式专项（选择题/古诗/翻译，C-Eval 28.83% → 41.31%）
  → SFT 两 epoch（7768 步，人格语料）
  → 身份小灶（300 步）
  → DPO 抛光（100 步）
```

## 核心发现

- **接口伤，不是知识伤**：注意力→KDA 转换后 C-Eval 一度跌至 ~28%，但四置换诊断证明模型"认识答案，只是不认识 A/B/C/D"。专项格式训练后恢复至 41%+，无需重训知识。
- **人格可以白嫖**：全程 15 个 C-Eval 守护点确认人格 SFT 未造成灾难性遗忘。
- **on-policy DPO 一轮即甜区**：第二轮迭代只增生复读，没有收益。

## 当前状态

- [x] 训练全部完成，最终权重已归档（C-Eval 41.83% ± 1.33）
- [ ] 技术报告 / 论文（workshop 版：接口伤诊断学）
- [ ] 训练代码与配方公开
- [ ] 权重发布（HuggingFace）
- [ ] 本地部署指南（AstrBot + HybridKDACache）

## License

代码与权重：Apache 2.0（基座 Qwen3-0.6B 为 Apache 2.0）

---

*QOSP = Qingyi Open Source Project，开源清漪计划。隶属于 DT-Project（数字孪生，Digital Twin）。*
