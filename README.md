# open-qingyi — QOSP（Qingyi Open Source Project / 开源清漪计划）

> 把全注意力模型"做手术"转成 KDA 混合架构，再把能力和人格一点点训回来的完整配方——三代手术、五次范式迭代、全部失败与诊断记录同步公开。

**QINGYI-KDA-0.6B**：以 Qwen3-0.6B-Base 为基座，将 28 层全注意力中的 21 层替换为 KDA（Kimi Delta Attention）线性注意力层，保留 7 层原始 GQA，经分层对齐、端到端知识蒸馏、格式专项、人格 SFT 与 DPO 抛光得到的混合架构模型（657.5M 参数）。

- 权重：[huggingface.co/shiershuihesaixiliya/qingyi-kda-0.6b](https://huggingface.co/shiershuihesaixiliya/qingyi-kda-0.6b)
- 技术报告（v1→v3 全记录）：[TECH_REPORT.md](TECH_REPORT.md)
- 逐步复现配方：[RECIPE.md](RECIPE.md)

## 相关工作坐标

注意力→线性注意力的转换并不新鲜，本项目在两个公开坐标之间：

- **GenDistill**（arXiv:2603.26556）已公开完成 Qwen3-0.6B→Hybrid-KDA（Stage 3a/3b、completion-only KD）。本项目是其配方族在消费级/租用单卡预算下的独立工程实现（解冻范围直接采用其 MLP 冻结消融结论），贡献在于其未覆盖的**接口伤四置换诊断学**、**人格对齐扩展**与全流程事故记录；
- **HALO**（THUNLP）的公开记录中，Qwen3→KDA 在其 Appendix B 同配置实验中 Stage 2 梯度发散失败。本管线端到端收敛；注意 HALO 正式管线同样保留 25% 注意力层，收敛差异的候选解释（门控初始化 g6、FP32 master、梯度裁剪、canonical hash 锁定）未经消融证明，仅作假设列出。

## 血缘链

```
Qwen3-0.6B-Base（C-Eval 50.59%，同口径教师基线）
  → v3 手术（21×KDA + 7×GQA，门控 g6，canonical hash 锁定）
  → Stage 2 分层对齐（hidden-state MSE，best step-900）
  → Stage 3a 端到端 forward-KL 蒸馏（7,000 步 / 57.3M tokens，T=1）
  → Stage 3b 格式专项（选择题四置换/古诗/翻译，1,000 步 / 8.2M packed tokens）
  → SFT 两 epoch（7,768 步，人格语料）
  → 身份小灶（300 步）
  → DPO 抛光（100 步，on-policy）
```

## 关键数字

| 阶段 | C-Eval | 备注 |
|---|---|---|
| 基座 Qwen3-0.6B-Base | 50.52% / 50.59%（两次同口径复测） | 教师 |
| v3 Stage 2 best | 25.48% | CE 已降至 4.1250，选择题仍躺随机线——"perplexity 会撒谎" |
| v3 Stage 3a step-7000 | 28.83% | valid CE gap +0.1277 |
| v3 Stage 3b step-8000 | **41.31%** | 接口修复，+12.48pt |
| 最终（DPO step-100） | **41.83% ± 1.33** | lm-eval ceval-valid，0-shot |

全部评测口径、误差带与 caveat 见 [TECH_REPORT.md](TECH_REPORT.md) §9/§10。

## 核心发现

- **接口伤是低分的主要可修复因素之一**：转换后 C-Eval 一度 ~28%，四选项置换诊断显示模型严重"黏住 A/B/C/D 标签"（选 A 率 81.06%，106/161 题四次换位仍黏同一标签）。1,000 步格式专项后选 A 率降至 58.5%、C-Eval 恢复 12.48pt。这是接口失能的强证据，**但不排除并存的知识损伤**（古诗/翻译仍有教师对而学生错的残例）。
- **人格可以"白嫖"**：15 个 C-Eval 守护点跟踪下，人格 SFT/小灶/DPO 全程无灾难性遗忘（SFT 首点曾下探 38.63%，随后恢复并稳定在 39%~41%）。
- **on-policy DPO 一轮即甜区**：held-out reward acc 0.625→0.792；第二轮迭代只增生复读，整轮废弃。
- **bf16 会吞更新**：norm 类权重在 bf16 参数+动量下数百步不动，Stage 2/3 全部改用 FP32 master 才恢复有效学习。

## 仓库结构

```
qingyi_kda/     模型包：QingyiKDAConfig / 建模代码 / HybridKDACache（增量缓存）
scripts/        92 个训练与审计脚本：手术 v2/v3、Stage2 对齐、KD、格式专项、
                SFT、小灶、DPO、四置换诊断、canary 提取测试等
RECIPE.md       逐步复现配方
TECH_REPORT.md  技术报告（v1/v2/v3 全实验记录，含失败与事故）
LICENSE         Apache 2.0
```

## 快速开始

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "shiershuihesaixiliya/qingyi-kda-0.6b"
tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    REPO, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda")
model.eval()

prompt = "<|im_start|>user\n哥哥: 你是谁？<|im_end|>\n<|im_start|>assistant\n"
ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
out = model.generate(ids, max_new_tokens=80, do_sample=False,
                     pad_token_id=tok.eos_token_id)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

要求：CUDA GPU（KDA 层为 Triton kernel，无 CPU 回退）、`transformers>=5.0`、`fla-core`、`einops`、`triton==3.6.0`（3.7.x 会使 `chunk_kda` 训练假死）。采样部署建议 temperature ≤0.5（高温会冲散身份绑定）。

## 当前状态

- [x] 训练全部完成，最终权重已归档（C-Eval 41.83% ± 1.33，SHA-256 `f375a99f…78e51a`）
- [x] 训练代码与完整配方公开（[RECIPE.md](RECIPE.md)）
- [x] 权重发布（HF，含自定义建模代码，`trust_remote_code`）
- [x] 技术报告公开（[TECH_REPORT.md](TECH_REPORT.md)）
- [ ] 论文（workshop 版：接口伤诊断学；毕业论文版：三代手术全记录）
- [ ] 本地部署指南（AstrBot + HybridKDACache）

**隐私说明**：`scripts/` 中的身份探针示例使用占位符（`123456789`）；发布权重中的身份设定含真实号码，系作者有意为之（见模型卡）。原始聊天数据不公开——`data/` 涉及群友隐私，仅公开数据配方与构建脚本。

## License

代码与权重：Apache 2.0（基座 Qwen3-0.6B 为 Apache 2.0）。第三方训练数据集（fineweb-edu、IndustryCorpus2、SmolTalk2、COIG-CQIA 等）许可以各自数据源为准。

---

*QOSP = Qingyi Open Source Project，开源清漪计划。隶属于 DT-Project（数字孪生，Digital Twin）。*
