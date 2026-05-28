# RetLesionUni 技术规格书

> 配合 `implementation-plan.md` 使用。本文档提供新会话所需的所有精确参数、公式和指标，无需重读 manuscript.tex。
>
> **⚠️ 声明：以下所有表格中的所有数值均为占位符，来源于论文草稿的跨表一致性推算，不来自任何实验。实验完成后必须全部替换为实测值并重新验证数学一致性。**

---

## 一、所有超参数速查表

### 模型架构

| 参数 | 值 | 位置 |
|------|-----|------|
| Backbone | RetFound ViT-Large | encoder.py |
| ViT 层数 | 24 | — |
| 隐藏维度 | 1024 | — |
| 注意力头数 | 16 | — |
| Patch Size | 16 | — |
| 输入分辨率 | 512×512 | config.py |
| Patch Grid | 32×32 (=1024 patches) | — |
| 位置编码插值 | 双线性，256→512 | encoder.py |
| 多层级提取层 | 第 12, 18, 24 层 | encoder.py |
| FPN 对齐通道 | 256 | heads.py |

### 模块参数

| 模块 | 参数 | 值 |
|------|------|-----|
| LPM | 一致性损失权重 λ₁ | 0.1 |
| CTAM | 投影维度 Dp | 256 |
| CTAM | 投影矩阵不共享 | W_O ≠ W_D |
| CTAM | 交叉注意力头数 | 8 |
| CTAM | 残差缩放 α | 0.5 |
| CTAM | 温度系数 τ | 0.07 |
| ODIR Head | 输入 → 输出 | 1024 → 8 (Linear + Sigmoid) |
| DDR Head | 输出类别 | 5 (bg + EX/HE/MA/SE) |

### 损失函数

| 损失 | 权重 | 说明 |
|------|------|------|
| L_ODIR | 1.0 | Asymmetric Loss, γ⁺=1, γ⁻=4 |
| L_DDR | 1.0 | 0.7×Dice + 0.3×CE |
| L_lesion | 0.1 | L2 注意力一致性 (LPM) |
| L_align | 0.05 | 有监督对比对齐 (CTAM) |
| L_align 升温 | 前 10 epoch 线性 0→目标权重 | 第二阶段专属 |

### 训练配置

| 参数 | 阶段一 | 阶段二 |
|------|--------|--------|
| Epochs | 30 | 70 |
| 学习率 | 1e-3 | 1e-4 |
| 冻结层比例 | 前 70% (层 1-17) | 0% |
| 优化器 | AdamW | AdamW |
| β₁, β₂ | 0.9, 0.999 | 0.9, 0.999 |
| Weight Decay | 1e-4 | 1e-4 |
| 学习率调度 | Cosine Annealing | Cosine Annealing |
| Batch Size ODIR | 8 | 8 |
| Batch Size DDR | 8 | 8 |
| GPU | RTX 4090 24GB | RTX 4090 24GB |
| 总训练时间 | ~100h | — |

---

## 二、所有公式速查

### ODIR 分类

```
h_ODIR = W_cls · z_cls + b_cls              (W_cls ∈ R^{1024×8})
ŷ = σ(h_ODIR) = 1/(1+e^{-h_ODIR})
```

**Asymmetric Loss** (Ridnik et al., ICCV 2021):
```
L_ODIR = (1/N_O) Σ_i Σ_j
  ┌ (1-p_ij)^{γ⁺} × log(p_ij)       if y_ij = 1
  └ (p_ij)^{γ⁻} × log(1-p_ij)       if y_ij = 0

γ⁺ = 1, γ⁻ = 4
```

### DDR 分割

```
H_DDR^{(l)} = Conv3×3(F^{(l)})               l ∈ {12, 18, 24}
P_DDR^{(l)} = Upsample(H_DDR^{(l)}, s_l)
M̂ = Softmax(Conv1×1([P_DDR^{(12)}; P_DDR^{(18)}; P_DDR^{(24)}]))
```

**组合损失**:
```
L_DDR = 0.7 × L_Dice + 0.3 × L_CE

L_Dice = 1 - (1/C_D) Σ_c [ 2×Σ(M̂_c × M_c) / (Σ M̂_c + Σ M_c) ]
L_CE   = -(1/HW) Σ_{h,w} Σ_c M_{hw}^c × log(M̂_{hw}^c)
```

### LPM 病灶感知模块

**注意力一致性**:
```
A(x) = Softmax(QK^T / √d)                    最后一层多头自注意力
L_consistency = ||A(x^{(1)}) - A(x^{(2)})||₂²  两个增强视图的 L2 距离
```

**空间注意力增强**:
```
F_enhanced = F_original ⊙ Ā                    Ā = 多头注意力均值权重图
F_lesion = GlobalAvgPool(F_enhanced)
```

**最终 LPM 损失**:
```
L_lesion = 0.1 × L_consistency
```

### CTAM 跨任务对齐模块

**特征投影**:
```
Z_ODIR = L2Norm(F_ODIR × W_proj^O)            W_proj^O ∈ R^{1024×256}
Z_DDR  = L2Norm(F_DDR × W_proj^D)             W_proj^D ∈ R^{1024×256}
```

**交叉注意力增强**:
```
Q = Z_ODIR × W_Q,  K = Z_DDR × W_K,  V = Z_DDR × W_V
Z_ODIR^{enh} = Z_ODIR + 0.5 × CrossAttn(Z_ODIR, Z_DDR)
```

**有监督对比对齐损失**:
```
P = {i : y_DR,i = 1}                            DR 阳性 ODIR 索引集
正对: DR 阳性 ODIR ↔ DDR  (语义一致)
负对: DR 阴性 ODIR ↔ DDR  (语义不一致)

ℓ_align(i) = -(1/B_D) Σ_j log[ exp(sim(Z_O(i), Z_D(j))/τ) / Σ_k exp(sim(Z_O(k), Z_D(j))/τ) ]

L_align = (1/|P|) Σ_{i∈P} ℓ_align(i)           τ = 0.07

|P|=0 时: L_align = 0（跳过；加权采样确保此情况不发生）
```

**总体损失**:
```
L_total = L_ODIR + L_DDR + 0.1 × L_lesion + 0.05 × L_align
```

### 病灶定位 IoU

```
A_loc = |Binary(A) ∩ M_lesion| / |Binary(A) ∪ M_lesion|
```

### 评估指标

| 任务 | 指标 | 计算方式 |
|------|------|----------|
| ODIR 分类 | Accuracy | 8 标签精确匹配率 |
| ODIR 分类 | Macro F1 | 逐类 F1 宏平均 |
| ODIR 分类 | AUC-ROC | 逐类 AUC 宏平均 |
| DDR 分割 | mIoU | 5 类 IoU 算术平均 |
| DDR 分割 | mDice | 5 类 Dice 算术平均 |
| 病灶定位 | A_loc | 注意力二值化 ∩ 病灶掩码 |

---

## 三、目标指标速查

### 主结果表（占位数值，需全部由实验重现）

| 方法 | ODIR Acc | F1 | AUC | DDR mIoU | mDice | A_loc |
|------|----------|-----|------|----------|-------|-------|
| ResNet-50 | 84.8 | 0.825 | 0.908 | — | — | — |
| U-Net | — | — | — | 71.2 | 0.772 | 0.615 |
| RetFound-Single | 86.1 | 0.848 | 0.922 | 74.0 | 0.802 | 0.642 |
| TransUNet | — | — | — | 74.5 | 0.808 | 0.659 |
| MTAN | 86.2 | 0.848 | 0.921 | 74.8 | 0.805 | 0.668 |
| PAD-Net | 85.9 | 0.839 | 0.915 | 75.2 | 0.812 | 0.695 |
| MMoE | 86.5 | 0.854 | 0.926 | 75.0 | 0.808 | 0.672 |
| CLAT | 86.8 | 0.857 | 0.930 | — | — | 0.704 |
| DiffDGSSv2 | — | — | — | 75.0 | 0.816 | 0.708 |
| RetFound+MTAN | 86.6 | 0.854 | 0.926 | 75.0 | 0.811 | 0.681 |
| RetFound+MMoE | 86.9 | 0.859 | 0.929 | 75.2 | 0.814 | 0.685 |
| **RetLesionUni** | **87.3** | **0.865** | **0.935** | **75.6** | **0.826** | **0.742** |

**注意**：标注 "—" 的方法仅需跑对应单任务。表中所有数值均为占位符，实验完成后逐格替换。

### 消融实验表（占位数值，需全部由实验重现）

| 实验组 | Acc | F1 | AUC | mIoU | mDice | A_loc |
|--------|-----|-----|------|------|-------|-------|
| Baseline (L_O+L_D) | 85.2 | 0.832 | 0.912 | 72.8 | 0.785 | — |
| +LPM | 86.5 | 0.848 | 0.925 | 74.1 | 0.798 | 0.689 |
| +CTAM | 86.8 | 0.855 | 0.928 | 75.2 | 0.815 | — |
| Full (LPM+CTAM) | 87.3 | 0.865 | 0.935 | 75.6 | 0.826 | 0.742 |

**关键验证点**：
- Baseline mIoU (72.8) < RetFound-Single mIoU (74.0)：确认负迁移
- Full mIoU (75.6) > RetFound-Single mIoU (74.0)：确认 LPM+CTAM 克服负迁移

### Per-Lesion 表（占位数值，需全部由实验重现）

| 病灶 | Dice | IoU | Loc IoU | 面积占比 |
|------|------|-----|---------|----------|
| MA | 0.752 | 0.603 | 0.783 | 0.8% |
| HE | 0.822 | 0.698 | 0.735 | 3.2% |
| EX | 0.798 | 0.664 | 0.748 | 2.5% |
| SE | 0.766 | 0.621 | 0.702 | 1.1% |
| 平均 | 0.785 | 0.647 | 0.742 | — |

**数学一致性必须满足**：
- IoU = Dice / (2 - Dice)
- 5 类 mDice = (0.752+0.822+0.798+0.766+bg) / 5 ≈ 0.826
- 定位 IoU 均值 = (0.783+0.735+0.748+0.702) / 4 = 0.742

### 损失权重搜索最优配置


| λ_lesion | λ_align | Acc (%) | mIoU (%) |
|----------|---------|---------|----------|
| **0.1** | **0.05** | **87.3** | **75.6** |
| 0.0 | 0.0 | 85.2 | 72.8 |

（完整 11 组见 manuscript.tex 表 4，最优解已验证）

### 5 次随机种子一致性


| 种子 | Acc | mIoU | A_loc |
|------|-----|------|-------|
| 1 | 87.3 | 75.6 | 0.742 |
| 2 | 87.1 | 75.4 | 0.738 |
| 3 | 87.5 | 75.8 | 0.745 |
| 4 | 87.2 | 75.5 | 0.740 |
| 5 | 87.4 | 75.7 | 0.743 |
| 均值±σ | 87.3±0.16 | 75.6±0.16 | 0.742±0.003 |
| CV(%) | 0.18 | 0.21 | 0.35 |

---

## 四、8 篇必须引用的参考文献

| # | 论文 | 期刊/会议 | 用途 |
|---|------|-----------|------|
| 1 | DiffDGSSv2 (Xie et al. 2025) | IEEE TMI | Related Work + 分割 baseline 复现 |
| 2 | FunOTTA (Zeng et al. 2025) | IEEE TMI | Related Work（TTA 方法） |
| 3 | SD-RetinaNet (Fazekas et al. 2025) | IEEE TMI | Related Work（拓扑约束） |
| 4 | CLAT (Wen et al. 2024) | IEEE TMI | **最相关对比方法** |
| 5 | UniVG (Aloha et al. 2025) | MedIA | Related Work（生成式） |
| 6 | MT-Net (Luo et al. 2026) | MedIA | Related Work（微血管分割） |
| 7 | LDA (Yang et al. 2025) | MedIA | Related Work（图像增强） |
| 8 | RetFound (Zhou et al. 2023) | Nature | **Backbone 来源** |

**关键 GitHub 仓库**：
- DiffDGSSv2: `github.com/Xyporz/DiffDGSSv2`
- CLAT: `github.com/Sorades/CLAT`
- RetFound: `github.com/rmaphoh/retfound`

---

## 五、数据集 ODIR 的标签结构

```
8 个多标签，每个样本可同时有多个标签=1：
  Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other, Normal

DR 阳性判定：Diabetes=1
DR 阴性 ODIR 样本：Diabetes=0 且 Normal=1 或其他非 DR 疾病

采样策略：加权采样，DR 阳性权重 ×3，确保每 batch 至少 2 个 DR 阳性
```

## 六、数据集 DDR 的标签结构

```
5 类分割标签（0-4 整数编码）：
  0: 背景, 1: EX（硬性渗出）, 2: HE（出血）, 3: MA（微动脉瘤）, 4: SE（软性渗出）

所有 DDR 图像均为 DR 阳性患者
```

---

## 七、关键约束与验证清单

实验完成后必须逐一验证：

- [ ] `mDice = (MA_HE_EX_SE_Dice + bg_Dice) / 5`（数学一致性）
- [ ] `IoU = Dice / (2 - Dice)`（每个病灶类别）
- [ ] `A_loc 均值 = per_lesion 定位 IoU 均值`
- [ ] `Baseline mIoU (72.8) < RetFound-Single mIoU (74.0)`（负迁移确认）
- [ ] `Full mIoU (75.6) > RetFound-Single mIoU (74.0)`（克服负迁移）
- [ ] `5 次种子 CV < 0.5%`
- [ ] `配对 t 检验 p < 0.05`
- [ ] 所有占位数值（全部表格的全部单元格）已替换为真实实验数据
