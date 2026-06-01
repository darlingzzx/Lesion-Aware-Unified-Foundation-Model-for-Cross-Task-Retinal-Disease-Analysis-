# RetLesionUni 代码审查：剩余问题

> 2026-06-01 | 审查范围：heads.py, retlesionuni.py, lpm.py, trainer.py, ctam.py, encoder.py

---

## 🔴 P0 — 致命问题

### 问题 1：LPM 注意力权重 ≈ 0.001，增强特征被清零

**位置**: [lpm.py:64-66](src/retlesionuni/models/lpm.py#L64-L66)

```python
attn_mean = (attn1_avg + attn2_avg) / 2.0        # (B, N, N), 每行 softmax sum=1
attn_weights = attn_mean.mean(dim=1)               # (B, N), ≈ 1/N = 1/1024 ≈ 0.001
enhanced = patch_tokens * attn_weights.unsqueeze(-1)  # ❌ 特征被衰减 ~1000x
```

**根因**: ViT 注意力矩阵每行经 softmax 后和为 1。`mean(dim=1)` 对查询维度求均值，得到每个 key patch 接收的平均注意力 — 值恒为 1/N ≈ 0.001（均匀注意力）或接近此值（结构化注意力）。

**实测验证**（GPU）:
```
Attention weights mean = 0.000976  (理论值 1/1024 = 0.000977)
patch_tokens std: 0.0500  →  enhanced std: 0.00004877
Attenuation: 0.001x  ← 特征被压缩 1000 倍
```

**后果**: `f_odir_enhanced.mean(dim=1)` 池化后 ≈ 0，拼接到 CLS token 后 patch 分支贡献为 0。**这导致缺陷 1（LPM 特征流入分类）的修复完全无效。**

**修复方向**: 归一化注意力权重使其均值为 1：
```python
attn_weights = attn_mean.mean(dim=1)                     # (B, N), ≈ 0.001
attn_weights = attn_weights / (attn_weights.mean(dim=1, keepdim=True) + 1e-8)  # mean→1.0
enhanced = patch_tokens * attn_weights.unsqueeze(-1)     # ✅ 加权但不衰减
```

---

### 问题 2：Stage 2 验证时 ODIR 缺少 CTAM 融合项

**位置**: [trainer.py:338-343](src/retlesionuni/train/trainer.py#L338-L343) vs [retlesionuni.py:151-153](src/retlesionuni/models/retlesionuni.py#L151-L153)

```python
# 训练时 (retlesionuni.py:152-153):
ctam_fused = self.ctam_fusion(z_odir_enhanced)   # Linear(256→2048)
f_odir_for_cls = f_odir_for_cls + ctam_fused      # ✅ CTAM 项注入

# 验证时 (trainer.py:341-343):
f_odir = torch.cat([enc["cls_token"], patch_pooled], dim=-1)  # (B, 2048)
preds = self.model.odir_head(f_odir)              # ❌ 缺少 CTAM 项
```

**根因**: 验证时不经过 CTAM（无 DDR 特征可用），但 ODIR 头在训练时已学会依赖 CTAM 提供的偏置。验证时偏置消失 → 预测分布偏移。

**后果**: Stage 2 的 ODIR F1 不升反降（从 0.164 降到 0.116），因为训练和验证的特征分布不同。

**修复方向**: 两种方案 —
- **方案 A**: 验证时也通过 CTAM（需要 DDR batch），但这会改变验证流程
- **方案 B**: 训练时也将 CTAM 融合项做 dropout/perturb，让头不依赖它
- **方案 C**: 将 CTAM 融合改为残差风格的小量修正（如 `+ 0.1 * ctam_fused`），减少对验证的影响

---

## 🟡 P1 — 重要问题

### 问题 3：CTAM 特征融合未做 warmup

**位置**: [retlesionuni.py:151-153](src/retlesionuni/models/retlesionuni.py#L151-L153)

```python
# loss_align 有 warmup:
loss_align * self.lambda_align * self.warmup_factor  # warmup 从 0→1

# 但 ctam_fusion 没有:
ctam_fused = self.ctam_fusion(z_odir_enhanced)       # ❌ 全量注入，无 warmup
f_odir_for_cls = f_odir_for_cls + ctam_fused
```

**根因**: `loss_align` 通过 `warmup_factor` 逐 epoch 增大，但特征融合一直全量进行。Stage 2 前 3 个 epoch（warmup 阶段），CTAM 对齐尚未收敛，注入的是噪声。

**后果**: Stage 2 早期训练不稳定，`L_align` 振荡大（0.002 ~ 0.109）。

**修复方向**: 特征融合也乘以 `warmup_factor`：
```python
f_odir_for_cls = f_odir_for_cls + self.warmup_factor * ctam_fused
```

---

### 问题 4：ODIR 验证时 encoder 走不同代码路径

**位置**: [trainer.py:340](src/retlesionuni/train/trainer.py#L340) vs [retlesionuni.py:119](src/retlesionuni/models/retlesionuni.py#L119)

```python
# Stage 2 训练时:
enc1 = self.encoder(x_odir_v1, return_attention=self.use_lpm)  # return_attention=True

# 验证时:
enc = self.model.encoder(batch["image_v1"])  # return_attention=False (默认)
```

**根因**: 当 `return_attention=True` 时，encoder 的最后一层走自定义 `_block_forward_with_attention`（手动实现 attention + MLP）；`return_attention=False` 时走 timm 内置 `block(x)`。两者数学等价但浮点运算顺序不同。

**后果**: 验证时 encoder 输出的 `cls_token` 和 `patch_tokens` 与训练时来自不同计算路径，特征可能有微小差异。叠加问题 2 后，验证特征与训练特征有双重不一致。

**修复方向**: 验证时也传 `return_attention=True`（即使不使用返回的 attention），或统一使用同一代码路径。

---

### 问题 5：CTAM 融合层设计缺陷

**位置**: [retlesionuni.py:65](src/retlesionuni/models/retlesionuni.py#L65)

```python
self.ctam_fusion = nn.Linear(cfg.ctam.proj_dim, cfg.hidden_dim * 2)  
# Linear(256, 2048) = 526K 参数
```

**三个子问题**:

1. **参数量过大**: 526K 参数仅用于特征融合，占 ODIR 头（1.05M）的 50%
2. **尺度不匹配**: CTAM 输出 `z_odir_enhanced` 经 L2-normalize 后在单位超球面（每个元素 ≈ ±0.06），encoder 特征值范围 ±0.1~1.0，直接相加没有意义
3. **无门控**: 直接 `f + ctam_fused` 没有可学习的融合比例

**修复方向**: 使用轻量门控融合：
```python
self.ctam_gate = nn.Sequential(
    nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2048), nn.Sigmoid()
)
# 或者更简单：直接用小比例加法
f_odir_for_cls = f_odir_for_cls + 0.1 * self.ctam_fusion(z_odir_enhanced)
```

---

## 🟢 P2 — 次要问题

### 问题 6：LPM L1 loss 过小，正则化失效

**位置**: [lpm.py:90](src/retlesionuni/models/lpm.py#L90)

```python
loss = F.l1_loss(attn1, attn2)  # ≈ 2e-4
```

乘以 `lambda_lesion=0.1` 后 ≈ **2e-5**，占总 loss（~0.7）的 **0.003%**。LPM 对训练几乎无任何约束效果。

**修复方向**: 增大 `lambda_lesion`（如 10.0），将 loss 拉升到 ~2e-3，占总 loss ~0.3%。

---

### 问题 7：LPM docstring 过时

**位置**: [lpm.py:1-8](src/retlesionuni/models/lpm.py#L1-L8)

文档仍描述 "cosine distance" 和 MSE，但默认已改为 L1。

---

### 问题 8：ODIRClassifier BatchNorm1d 在极小 batch 下不稳定

**位置**: [heads.py:23](src/retlesionuni/models/heads.py#L23)

```python
nn.BatchNorm1d(hidden)  # hidden=512
```

训练时 batch_size=6，BatchNorm 统计量有噪声但可接受。但如果未来 batch_size=1（推理时），eval 模式下使用 running stats 没问题。

---

## 问题汇总

| # | 级别 | 位置 | 一句话 | 后果 |
|---|------|------|--------|------|
| 1 | **P0** | lpm.py:64-66 | 注意力权重 ≈ 0.001，增强特征清零 | 缺陷1修复完全无效 |
| 2 | **P0** | trainer.py:341-343 | Stage2 验证缺 CTAM 融合 | ODIR F1 训练/验证分裂 |
| 3 | **P1** | retlesionuni.py:152 | CTAM 融合无 warmup | Stage2 早期注入噪声 |
| 4 | **P1** | trainer.py:340 | 验证 encoder 路径不一致 | 双重特征分布偏移 |
| 5 | **P1** | retlesionuni.py:65 | CTAM 融合 526K 参数 + 尺度不匹配 | 资源浪费，融合不合理 |
| 6 | **P2** | lpm.py:90 | L1 loss ≈ 2e-4，正则失效 | LPM 对训练无约束 |
| 7 | **P2** | lpm.py:1-8 | docstring 未更新 | 误导 |
| 8 | **P2** | heads.py:23 | BatchNorm 小 batch | 轻微不稳定 |

---

## 修复优先级

| 顺序 | 问题 | 改动量 | 预期收益 |
|------|------|--------|----------|
| **1** | 问题 1: LPM 权重归一化 | 2 行 | LPM 增强特征真正生效 |
| **2** | 问题 3: CTAM 融合 warmup | 1 行 | Stage2 早期稳定 |
| **3** | 问题 5: CTAM 融合轻量化 | 小 | 减少参数 + 合理融合 |
| **4** | 问题 2: 验证 CTAM 一致 | 中 | ODIR 训练/验证对齐 |
| **5** | 问题 6: LPM lambda 调大 | 1 行 | LPM 正则有效 |
| **6** | 问题 4: encoder 路径一致 | 小 | 消除最后的不一致 |
| **7** | 问题 7: docstring | 小 | — |

---

## 相关文件

| 文件 | 涉及问题 |
|------|----------|
| `src/retlesionuni/models/lpm.py` | #1, #6, #7 |
| `src/retlesionuni/models/retlesionuni.py` | #2, #3, #5 |
| `src/retlesionuni/train/trainer.py` | #2, #4 |
| `src/retlesionuni/models/heads.py` | #8 |
| `configs/default.yaml` | #6 |
