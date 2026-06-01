# RetLesionUni 设计缺陷与修复方案

> 2026-06-01 | 训练早停 | 最佳模型: `outputs/checkpoints/retlesionuni_full/resume_checkpoint.pth` (stage2 epoch5, DDR mDice=0.278)

---

## 当前状态

| 项目 | 值 |
|------|-----|
| 最佳 checkpoint | `resume_checkpoint.pth` (3.5GB, stage2 epoch 5) |
| DDR 测试 mDice | 0.314 (U-Net 基线 0.238, SOTA 0.50-0.68) |
| ODIR 测试 F1/AUC | 0.233 / 0.672 (SOTA 0.89-0.94) |
| MA/SE 病灶 | Dice=0 (完全未检测到) |
| 训练量 | 10 stage1 + 5 stage2 = 15/100 epochs |
| 配置 | batch_size=6, fp16, gradient_checkpointing=true, num_workers=0 |

---

## 缺陷 1（致命）：LPM 增强特征被丢弃

**位置**: [retlesionuni.py:113-121](src/retlesionuni/models/retlesionuni.py#L113-L121)

```python
if self.use_lpm:
    enc2 = self.encoder(x_odir_v2, return_attention=True)
    f_odir_enhanced, loss_lesion = self.lpm(
        enc1["patch_tokens"],
        enc1["last_attention"],
        enc2["last_attention"],
    )
else:
    f_odir_enhanced = enc1["patch_tokens"]

odir_logits = self.odir_head(enc1["cls_token"])  # ❌ 永远用 cls_token，f_odir_enhanced 被丢弃
```

**问题**: LPM 计算了注意力加权的增强 patch 特征 `f_odir_enhanced`（形状 `(B, N, 1024)`），但 ODIR 分类头只用 `enc1["cls_token"]`。增强特征被完全丢弃，LPM 的唯一作用是产生 `loss_lesion` 梯度。

**修复**:
1. 对 `f_odir_enhanced` 做全局平均池化（或加 CLS token）→ 得到增强的图像特征
2. 将增强特征送入 ODIR 分类头（替换或拼接原始 CLS token）
3. 同时将 `f_odir_enhanced` 传给 CTAM（代替 `enc1["cls_token"]`），让对齐模块也能受益

```python
# 建议修改
if self.use_lpm:
    enc2 = self.encoder(x_odir_v2, return_attention=True)
    f_odir_enhanced, loss_lesion = self.lpm(...)
    # 池化增强特征用于分类
    f_odir_pooled = f_odir_enhanced.mean(dim=1)  # (B, 1024)，全局平均池化
    f_odir_for_cls = f_odir_pooled
else:
    f_odir_for_cls = enc1["cls_token"]

odir_logits = self.odir_head(f_odir_for_cls)  # ✅ 使用增强特征
```

---

## 缺陷 2（致命）：CTAM 对齐特征被丢弃

**位置**: [retlesionuni.py:127-135](src/retlesionuni/models/retlesionuni.py#L127-L135)

```python
if self.use_ctam:
    dr_labels = odir["dr_label"].to(x_odir_v1.device)
    _, _, loss_align = self.ctam(
        enc1["cls_token"],
        enc_ddr["cls_token"],
        dr_labels,
    )
# ❌ CTAM 返回的 z_odir_enhanced (256维) 和 z_ddr (256维) 被丢弃
```

**问题**: CTAM 做了投影→L2归一化→交叉注意力→残差连接，产出了增强的对齐特征 `z_odir_enhanced`。但这些特征从未用于分类或分割。跨任务对齐对预测零影响。

**修复**:
1. 将 `z_odir_enhanced`（256维）与 ODIR 原始特征拼接或融合，送入分类头
2. 将 `z_ddr`（256维）与 DDR 的 CLS token 融合，注入分割头的解码过程
3. 或者更简单：用 `z_odir_enhanced` 作为 ODIR 分类的辅助输入

```python
# 建议修改
if self.use_ctam:
    z_odir_enhanced, z_ddr_aligned, loss_align = self.ctam(
        f_odir_for_cls,   # 来自 LPM 的增强特征
        enc_ddr["cls_token"],
        dr_labels,
    )
    # 融合对齐特征到分类
    f_odir_final = torch.cat([f_odir_for_cls, z_odir_enhanced], dim=-1)  # 1024+256=1280维
else:
    f_odir_final = f_odir_for_cls

odir_logits = self.odir_head(f_odir_final)
```

> 注意：ODIR 分类头输入维度需从 1024 改为 1280

---

## 缺陷 3（致命）：DDR 分割头 8 倍上采样导致小病灶丢失

**位置**: [heads.py:68-100](src/retlesionuni/models/heads.py#L68-L100)

```python
# 三层特征都上采样到 64×64
feat = F.interpolate(feat, size=(grid_size * 2, grid_size * 2), ...)  # 32→64
features.append(feat)

fused = torch.cat(features, dim=1)           # 64×64 拼接
out = self.final_conv(fused)                  # 64×64 上卷积
out = F.interpolate(out, size=(512, 512), ...) # ❌ 一步 8x 双线性到 512×512
```

**问题**: 有效分辨率仅 64×64，最后一步 8 倍双线性上采样。MA（微动脉瘤，几个像素）在 64×64 特征图上完全没有空间信息，双线性插值无法恢复。这就是 **MA Dice=0, SE Dice=0** 的直接原因。

**修复**: 采用渐进式上采样 + 跳跃连接（标准 U-Net/FPN 做法）

```python
# 建议：渐进式解码器
# Layer 24 (32×32) → upsample 2x → 64×64, concat with Layer 18
# Layer 18 (64×64) → upsample 2x → 128×128, concat with Layer 12  
# Layer 12 (128×128) → upsample 2x → 256×256
# Final conv → upsample 2x → 512×512 (最后一步最多 2x)

# 或者：直接用预训练的 CNN 解码器（如 DeepLabV3+ decoder），
# 将 ViT patch tokens reshape 为 2D 特征图后接入
```

具体方案（推荐）:
```python
class DDRSegmentator(nn.Module):
    def __init__(self, dim_in=1024, num_classes=5):
        # 渐进式解码：32→64→128→256→512
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(dim_in, 512, 2, 2),  # 32→64
            nn.BatchNorm2d(512), nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512 + 512, 256, 2, 2),  # 64→128, +skip
            nn.BatchNorm2d(256), nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, 2, 2),  # 128→256, +skip
            nn.BatchNorm2d(128), nn.ReLU()
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2),  # 256→512
            nn.BatchNorm2d(64), nn.ReLU()
        )
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, patch_tokens_list, img_size=512):
        # tokens_list: [layer12(B,N,1024), layer18(B,N,1024), layer24(B,N,1024)]
        # 从高层到低层逐步上采样
        ...
```

---

## 缺陷 4（重要）：ODIR 分类头过于简单

**位置**: [heads.py:12-32](src/retlesionuni/models/heads.py#L12-L32)

```python
class ODIRClassifier(nn.Module):
    def __init__(self, dim_in=1024, num_classes=8):
        self.fc = nn.Linear(dim_in, num_classes)  # 仅一层 Linear！

    def forward(self, cls_token):
        return torch.sigmoid(self.fc(cls_token))
```

**问题**: 307M 参数的 ViT-Large 编码器后面只接了一个 Linear(1024, 8) = 8K 参数。对比 SOTA（2-3 层 MLP + Dropout + BatchNorm + 注意力池化），过于简陋。CLS token 是全局表示，无法关注局部病灶。

**修复**: 至少 2 层 MLP + Dropout
```python
class ODIRClassifier(nn.Module):
    def __init__(self, dim_in=1024, num_classes=8, hidden=512, dropout=0.3):
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):
        return torch.sigmoid(self.mlp(x))
```

> 如果修复了缺陷 1+2，输入维度会变化（如 1280），需同步调整。

---

## 缺陷 5（中等）：LPM 注意力损失数值过小

**位置**: [lpm.py:50-52](src/retlesionuni/models/lpm.py#L50-L52)

**已部分修复**（从 `F.mse_loss` 改为 `reduction='none' + sum + mean`），但仍有问题：

**问题**: 注意力图元素 ≈ 1/N = 1/1024 ≈ 0.001。两层增强视图的注意力通常高度相似（差值 ~1e-5），平方后 ~1e-10，sum 后约 0.1。乘以 `lambda_lesion=1.0` 和 warmup factor 后约 0.05 — 但对总 loss=0.7 来说占比 <10%。

**修复**: 改用对注意力差异更敏感的度量
```python
# 方案 A: L1 loss（对微小差异更敏感）
loss_consistency = F.l1_loss(attn1_avg, attn2_avg)

# 方案 B: 余弦距离
cos_sim = F.cosine_similarity(attn1_avg.flatten(1), attn2_avg.flatten(1))
loss_consistency = (1 - cos_sim).mean()

# 方案 C: KL 散度（把注意力视为概率分布）
loss_consistency = F.kl_div(
    attn1_avg.log(), attn2_avg, reduction='batchmean'
)
```

---

## 缺陷 6（中等）：训练/验证特征路径不一致

**位置**: [trainer.py:271-284](src/retlesionuni/train/trainer.py#L271-L284)

```python
# 训练时 (forward)
enc1 = self.encoder(x_odir_v1, return_attention=self.use_lpm)  # return_attention=True
if self.use_lpm:
    enc2 = self.encoder(x_odir_v2, return_attention=True)

# 验证时 (_validate_odir)
enc = self.model.encoder(batch["image_v1"])  # return_attention=False（默认）
preds = self.model.odir_head(enc["cls_token"])
```

**问题**: 训练时 `return_attention=True` 会改变 encoder 的执行路径（最后一层走 `_block_forward_with_attention`），验证时走 `block(x)`。两者行为不完全一致。

**修复**: 验证时也设置 `return_attention=True`，或训练时也走 `block(x)` 路径取注意力。

---

## 缺陷 7（中等）：ODIR 分类只用 CLS token，忽略 patch 级特征

**位置**: [retlesionuni.py:121](src/retlesionuni/models/retlesionuni.py#L121)

**问题**: ODIR 疾病分类（糖尿病视网膜病变、青光眼、白内障等）中，部分疾病（如白内障）表现为全局图像变化，适合用 CLS token；但糖尿病视网膜病变表现为局部病灶（微动脉瘤、出血等），应该关注 patch 级局部特征。当前只用 CLS token 缺少局部信息。

**修复**: 将 patch tokens 的全局池化结果与 CLS token 融合
```python
patch_pooled = enc1["patch_tokens"].mean(dim=1)  # (B, 1024)
cls_features = enc1["cls_token"]                   # (B, 1024)
fused = torch.cat([cls_features, patch_pooled], dim=-1)  # (B, 2048)
odir_logits = self.odir_head(fused)
```

---

## 修复优先级

| 顺序 | 缺陷 | 预期收益 | 改动量 | 状态 |
|------|------|----------|--------|------|
| **1** | DDR 头上采样 (缺陷 3) | MA/SE Dice 从 0→0.15+ | 中 | ✅ 已修复 |
| **2** | LPM 特征流入分类 (缺陷 1) | ODIR F1 +5-10% | 小 | ✅ 已修复 |
| **3** | CTAM 特征流入分类 (缺陷 2) | ODIR F1 +3-5% | 小 | ✅ 已修复 |
| **4** | ODIR 头加深 (缺陷 4) | ODIR F1 +5-10% | 小 | ✅ 已修复 |
| **5** | LPM 损失函数 (缺陷 5) | L_les 更稳定 | 小 | ✅ 已修复 |
| **6** | patch 特征融合 (缺陷 7) | 局部病灶检测 | 小 | ✅ 已修复 |
| **7** | 训练/验证一致性 (缺陷 6) | 验证更准确 | 小 | ✅ 已修复 |

---

## 其他注意事项

### 训练配置
- **config**: `configs/default.yaml`
- **num_workers 必须为 0**（Windows DataLoader 多进程死锁问题）
- **gradient_checkpointing=true** 可在 12GB 显存下跑 Stage 2
- **batch_size ≤ 6** 避免 OOM
- `resume: true` 支持断点续训

### 环境
- **Conda**: `dlenv`, Python `C:/Users/zhangjuntao/.conda/envs/dlenv/python.exe`
- **启动命令**: `KMP_DUPLICATE_LIB_OK=TRUE python scripts/train.py`
- **工作目录**: `D:\RetLesionUni\Lesion-Aware-Unified-Foundation-Model-for-Cross-Task-Retinal-Disease-Analysis-\`

### 最佳 checkpoint
- **路径**: `outputs/checkpoints/retlesionuni_full/resume_checkpoint.pth`
- **内容**: stage2 epoch 5, DDR mDice(验证)=0.278
- **大小**: 3.5GB
- 只有这一个 checkpoint 了（中间文件已清理，释放 ~21.5GB）

### 参考基准
- DDR mDice: U-Net 0.238, SOTA 0.50-0.68 (MCFNet 0.679)
- ODIR F1: SOTA 0.89-0.94 (DKCNet 0.943)
- 我们当前: DDR 0.314 / ODIR 0.233（15 epochs，15% 计划训练量）

---

## 相关文件索引

| 文件 | 说明 |
|------|------|
| `src/retlesionuni/models/retlesionuni.py` | 顶层模型，forward 逻辑（缺陷 1,2） |
| `src/retlesionuni/models/heads.py` | ODIR/DDR 头（缺陷 3,4,7） |
| `src/retlesionuni/models/lpm.py` | LPM 模块（缺陷 5） |
| `src/retlesionuni/models/ctam.py` | CTAM 跨任务对齐 |
| `src/retlesionuni/models/encoder.py` | ViT 编码器 + gradient checkpointing |
| `src/retlesionuni/train/trainer.py` | 训练循环 + resume 逻辑（缺陷 6） |
| `src/retlesionuni/train/checkpoint.py` | checkpoint 存取 |
| `src/retlesionuni/data/joint_loader.py` | 联合数据加载器 |
| `src/retlesionuni/losses/` | ASL/Dice/对比损失 |
| `src/retlesionuni/eval/metrics.py` | 评估指标计算 |
| `configs/default.yaml` | 训练配置 |

---

## 本项目已做修改记录

1. `encoder.py`: pos_embed 加载顺序修复 + gradient checkpointing 支持
2. `trainer.py`: resume 断点续训 + 每 epoch 保存 resume checkpoint
3. `checkpoint.py`: 新增 scheduler state / stage_name 保存恢复
4. `lpm.py`: MSE loss 改为 sum+mean（修复 L_les=0 问题）
5. `retlesionuni.py`: 传递 gradient_checkpointing 标志
6. `dice_loss.py`: target.long() 修复
7. `default.yaml`: epochs 30→10+20, save_every=2, batch_size=6, num_workers=0, gradient_checkpointing=true, lambda_lesion=1.0, resume=true
8. `heads.py`: DDR 头渐进式解码器（ConvTranspose2d 32→64→128→256→512 + skip connections，替代单步 8x 双线性）
9. `heads.py`: ODIR 分类头加深（Linear→MLP: 1024→512→8 + BatchNorm + Dropout）
10. `retlesionuni.py`: LPM 增强特征流入 ODIR 分类（f_odir_enhanced 池化后与 CLS token 融合）
11. `retlesionuni.py`: CTAM 对齐特征流入 ODIR 分类（z_odir_enhanced 拼接到分类特征）
12. `retlesionuni.py`: patch 级特征与 CLS token 融合（cls_token + patch_pooled → 2048 维）
13. `retlesionuni.py`: ODIR 头输入维度从 1024 改为 2304（cls 1024 + patch 1024 + ctam 256）
14. `lpm.py`: 损失函数从 MSE 改为余弦距离（默认），支持 L1/MSE 选项
15. `trainer.py`: _validate_odir 使用与训练一致的 cls+patch+ctam_pad 特征路径

**重要**: 由于 ODIR 头输入维度从 1024→2304，旧的 checkpoint 中 `odir_head` 权重不兼容。需要从头开始训练或使用新 checkpoint。
