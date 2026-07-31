# RECIPE — QINGYI-KDA-0.6B 完整配方

本文档记录从 Qwen3-0.6B-Base 到 QINGYI-KDA-0.6B 的完整复现配方。
所有数字均来自实际训练日志；脚本见 `scripts/`，核心包见 `qingyi_kda/`。

## 0. 环境

- GPU：32GB 级单卡（实测 24GB 可用，峰值 ~22GB）；Stage 2/3 亦可在 16GB 级以更小 micro-batch 运行
- `torch` + `fla-core` + `triton` + `transformers>=5.0` + `lm-eval`（评测）+ `bitsandbytes`（SFT 8bit Adam）
- 基座：`Qwen/Qwen3-0.6B-Base`（Apache 2.0）

## 1. 手术：注意力 → KDA 混合替换

```bash
python scripts/build_hybrid_v3.py   # 21×KDA + 7×GQA，gate 方案 g6
```

- 布局：每 4 层保留 1 层原生 GQA（3,7,11,15,19,23,27），其余 21 层替换为 KDA
- KDA 配置：hidden 1024 / 16 heads / head_dim 128 / short-conv 4
- 教师 Q/K/V 投影权重初始化实验：无收益（记录于案，默认关闭）

## 2. Stage 2 — 分层对齐（hidden-state MSE）

```bash
python scripts/align_layers.py --init models/qingyi-hybrid-init-v3-canonical \
  --total-steps 1000 --micro-batch 2 --grad-accum 2 --lr 1e-4
```

- 每层 KDA 的 hidden 输出对齐教师的对应层输出，教师冻结、误差 detach 不串层
- **必须 FP32 master 参数/动量**：bf16 直接 AdamW 会让 ~1.0 附近的 norm 权重更新被半 ULP 吞掉
- 数据游标（CursorMixer）支持逐 token 一致的中断恢复

## 3. Stage 3a — 端到端 KL 蒸馏

```bash
python scripts/kd_e2e.py --resume <stage2-best> --total-steps 8000 \
  --micro-batch 2 --grad-accum 2 --lr 2.5e-5 --temperature 1.0 \
  --expect-init-hash <sha256>
```

- 纯 forward KL（teacher‖student），T=1，alpha_CE=0，57M→65M tokens
- 解冻范围：KDA + 全部 MLP + block/final LayerNorm ≈ 457.9M（embedding/lm_head/保留 attention 冻结）
- 有效 batch 8192 tokens/step；绕过 CausalLM 完整 logits（直接取 last_hidden_state 过 lm_head）
- C-Eval 轨迹：转换后 ~28% → 8000 步 41.31%

## 3b. 格式专项（关键修复）

转换后 C-Eval ~28% 的根因诊断（四选项置换实验）：

- 模型 81% 预测选 A，选项换位后不跟随答案内容 → **接口伤，不是知识伤**
- 数据：6,250 道教师验证 MCQ ×4 置换（A/B/C/D 各 25%）+ 3,000 古诗问答 + 3,000 翻译
- 与评测集精确/5-gram 双重零污染验证
- completion-only KL（prompt mask=0），1000 步，C-Eval 28.83% → **41.31%**

## 4. SFT 人格训练

```bash
python scripts/sft.py --start-from <stage3b-best> --epochs 2 \
  --lr 1e-5 --micro-batch 8 --max-steps 7768
```

- 数据 ~31k 条（真实聊天清洗 + 合成身份 QA），held-out 200 条守护
- bf16 参数 + 8bit Adam 可训（norm 类权重更新会被吞，主体正常）
- **15 个 C-Eval 守护点全程 ≥39%**：人格训练未造成灾难性遗忘
- 注意 `--max-steps` 为累加语义；续 epoch 用 `--epochs 2` 而非 `--resume`

## 5. 身份小灶

```bash
python scripts/sft.py --dataset data/sft/boost_dataset.pt --start-from <sft-final> \
  --epochs 1 --lr 1e-5 --max-steps 300
```

- 8975 条 = 249 独特身份 QA ×20 + 4000 聊天；**300 步点燃，400 步漫溢**（事实题开始否认 H₂O、复读塌陷）——浓度安全窗口 ~300 步，每 100 步必须生成探针验收

## 6. DPO 抛光

```bash
python scripts/dpo_v2_sample.py   # on-policy：当前模型对坑题 T0.9 采样抓真实失败
python scripts/dpo_v2_build.py    # 组装 chosen/rejected pairs
python scripts/dpo.py --pairs data/dpo/v2_pairs.jsonl --start-from <boost-best> \
  --epochs 1 --lr 1e-6 --beta 0.1 --micro-batch 4
```

- 545 pairs（identity 246 + chat 299），held-out reward acc 0.625→0.792（step 100 best）
- policy+ref 双模型显存：micro-batch 4 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- **第二轮迭代无效**（acc 0.646 未回 0.792，复读增生）——一轮即甜区

## 7. 验收与终点

- 最终 C-Eval **41.83% ± 1.33**（教师基线 50.59%，同口径 lm-eval ceval-valid）
- 部署：T 0.3~0.5 / top_p 0.9 / rep_penalty 1.1（T≥0.7 身份绑定被采样冲散）
- 权重 SHA-256：`f375a99f057db4060be07a28d2beb2447ae5d9164c36652b0c0a84404f78e51a`

## 已知坑位（运维）

- bf16 直接 AdamW 吞更新 → FP32 master（Stage 2/3 必须）
- DPO/SFT checkpoint ≈ 2.5G/份，磁盘需预留
- KDA 不读 attention_mask：生成必须左 pad 或等长批
- resume 恢复数据游标需校验 seed/数据 SHA，防静默续错
