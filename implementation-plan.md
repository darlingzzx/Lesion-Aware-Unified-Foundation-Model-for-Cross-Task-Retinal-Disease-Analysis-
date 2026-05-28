# RetLesionUni 完整实现计划

> **状态说明**：论文草稿已按导师模板完成结构对齐和数据一致性审查。**⚠️ 论文中每一个数字（所有表格的所有单元格）均为占位数值，所有实验尚未执行，代码待从零实现。本文档为从零补全所有代码和实验的完整规划。**

---

## 零、论文速览

### 做什么

糖尿病视网膜病变（DR）眼底图像分析中，同时做两件事：
- **ODIR 数据集**上做多标签疾病分类（7 类疾病 + Normal = 8 标签）
- **DDR 数据集**上做像素级病灶分割（MA/HE/EX/SE 四种病灶）

### 核心创新

两个模块搭在共享 RetFound ViT-Large 编码器上：

1. **病灶感知模块（LPM）**：给同一张图做两次随机增强 → 提取两次注意力图 → 约束它们一致（`L2 loss`） → 用注意力图加码编码器特征
2. **跨任务对齐模块（CTAM）**：ODIR 特征和 DDR 特征 → 两个独立的投影矩阵映射到 256 维 → 交叉注意力增强 → 以 DR 标签做有监督对比对齐

### 实验规模

- ~400M 参数（RetFound ViT-Large）
- RTX 4090 × 1，24GB 显存
- 训练约 100 小时
- 8 个对比基线 + 消融实验 + 权重分析 + 5 次随机种子稳定性检验 + 可解释性分析

---

## 一、环境搭建

### 1.1 硬件

| 需求 | 说明 |
|------|------|
| GPU | 单张 NVIDIA RTX 4090 24GB（最低要求：24GB 显存） |
| RAM | ≥ 64GB |
| 存储 | ≥ 200GB（ODIR + DDR + RetFound 预训练权重 + 实验产物） |

### 1.2 软件环境

```bash
# 创建 conda 环境
conda create -n retlesionuni python=3.10 -y
conda activate retlesionuni

# PyTorch（CUDA 12.1）
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# 核心依赖
pip install transformers==4.44.0          # RetFound ViT 模型加载
pip install timm==0.9.16                  # ViT 模型工具
pip install opencv-python==4.10.0         # 图像预处理
pip install albumentations==1.4.8         # 数据增强
pip install scikit-learn==1.5.1           # 评估指标
pip install scipy==1.14.0                 # 统计检验
pip install pandas==2.2.2                 # 数据处理
pip install matplotlib==3.9.1             # 可视化
pip install seaborn==0.13.2               # 热力图
pip install tensorboard==2.17.0           # 训练监控
pip install tqdm==4.66.4                  # 进度条
pip install einops==0.8.0                 # 张量操作
pip install monai==1.3.2                  # 医学图像工具
```

### 1.3 目录结构

```
retlesionuni/
├── data/
│   ├── odir/                    # ODIR 数据集
│   │   ├── images/              # 约 7000 张眼底图像
│   │   └── labels.csv           # 多标签标注
│   ├── ddr/                     # DDR 数据集
│   │   ├── images/              # DR 患者眼底图像
│   │   └── masks/               # 像素级分割标注 (EX/HE/MA/SE)
│   └── idrid/                   # IDRiD 外部验证集（可选）
├── pretrained/
│   └── retfound.pth             # RetFound ViT-Large 预训练权重
├── src/
│   ├── config.py                # 全局配置
│   ├── dataset/
│   │   ├── odir_dataset.py      # ODIR 数据加载
│   │   ├── ddr_dataset.py       # DDR 数据加载
│   │   └── transforms.py        # 数据增强与预处理
│   ├── models/
│   │   ├── encoder.py           # RetFound ViT-Large 封装
│   │   ├── lpm.py               # 病灶感知模块
│   │   ├── ctam.py              # 跨任务对齐模块
│   │   ├── heads.py             # ODIR 分类头 + DDR 分割头
│   │   └── retlesionuni.py      # 整体框架组装
│   ├── losses/
│   │   ├── asl_loss.py          # Asymmetric Loss（ODIR）
│   │   ├── dice_loss.py         # Dice + CE Loss（DDR）
│   │   ├── consistency_loss.py  # 注意力一致性 Loss
│   │   └── contrastive_loss.py  # 有监督对比对齐 Loss
│   ├── train/
│   │   ├── trainer.py           # 训练循环（两阶段微调）
│   │   ├── scheduler.py         # 学习率调度
│   │   └── checkpoint.py        # 模型保存/恢复
│   ├── eval/
│   │   ├── metrics.py           # 评估指标计算
│   │   ├── statistical_test.py  # 配对 t 检验
│   │   └── explainability.py    # Attention Rollout + 病灶定位 IoU
│   └── utils/
│       ├── roi_extract.py       # ROI 提取 + 霍夫圆检测
│       ├── preprocess.py        # 数据预处理管线
│       └── logger.py            # 日志/TensorBoard
├── experiments/
│   ├── exp01_baseline/          # 主对比实验
│   ├── exp02_ablation/          # 消融实验
│   ├── exp03_weights/           # 损失权重分析
│   ├── exp04_seeds/             # 多随机种子
│   ├── exp05_per_lesion/        # 按病灶细粒度分析
│   ├── exp06_explainability/    # 可解释性分析
│   └── exp07_cross_domain/      # 跨域泛化
├── outputs/                     # 输出产物
│   ├── checkpoints/
│   ├── logs/
│   ├── figures/
│   └── tables/
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_result_analysis.ipynb
    └── 03_figure_generation.ipynb
```

---

## 二、数据准备（预计 3-5 天）

### 2.1 ODIR 数据集

```
来源：北京大学（Li et al., 2021）
规模：约 3,500 组患者 × 2 眼 ≈ 7,000 张图像
标签：8 类多标签分类
      Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other abnormalities, Normal
划分：70% / 15% / 15%（按患者级别划分，确保同一患者双眼在同一子集）
```

**待确认**：
- [ ] ODIR 数据集下载渠道（需联系原作者或 Kaggle）
- [ ] 图像原始分辨率和格式
- [ ] 标签文件的精确格式

**预处理步骤**（在 `src/utils/preprocess.py` 中实现）：

1. 剔除分辨率 < 256×256 的低质量图像
2. 霍夫圆检测提取视网膜 ROI
3. 缩放至 512×512
4. 数据增强（仅训练集）：
   - 随机旋转 ±30°
   - 水平/垂直翻转（p=0.5）
   - 亮度/对比度扰动 ±10%
   - Cutout 5%-10%

### 2.2 DDR 数据集

```
来源：南开大学（Li et al., 2019）
标签：像素级分割掩码，4 类病灶 + 背景
      EX（硬性渗出）, HE（出血）, MA（微动脉瘤）, SE（软性渗出）
特点：所有图像均为 DR 阳性患者
划分：按官方标准划分
```

**待确认**：
- [ ] DDR 数据集下载渠道
- [ ] 分割标注的文件格式（PNG mask / JSON / COCO？）
- [ ] 病灶标注的像素值编码规则

**预处理步骤**：

1. 同 ODIR 步骤 1-3
2. 分割 mask 在增强时同步变换（旋转/翻转与图像一致，Cutout 不应用于 mask）

### 2.3 IDRiD 数据集（外部验证，可选）

```
用途：跨域泛化零样本测试
标注：MA, HE, EX, SE 四种病灶，与 DDR 一致
状态：P3 优先级，可在主体实验完成后补充
```

### 2.4 RetFound 预训练权重

```
模型：RetFound ViT-Large（Zhou et al., Nature 2023）
下载：https://github.com/rmaphoh/retfound
参数：~300M（ViT-Large, 24 层, 1024 维, 16 头）
输入：原始 256×256，需要将位置编码插值至 512×512
```

**关键代码**——位置编码双线性插值：

```python
# src/models/encoder.py
import torch
import torch.nn.functional as F

def interpolate_pos_embed(pos_embed_old, new_grid_size=32):
    """
    pos_embed_old: (1, 257, 1024)  # 256×256 → 16×16 grid + CLS
    new_grid_size: 32  # 512×512 → 32×32 grid
    returns: (1, 1025, 1024)
    """
    cls_token = pos_embed_old[:, :1, :]  # CLS token
    pos_tokens = pos_embed_old[:, 1:, :]  # (1, 256, 1024)

    old_size = int(pos_tokens.shape[1] ** 0.5)  # 16
    pos_tokens = pos_tokens.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(pos_tokens, size=(new_grid_size, new_grid_size),
                                mode='bilinear', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_grid_size**2, -1)
    return torch.cat([cls_token, pos_tokens], dim=1)
```

---

## 三、模型实现（预计 10-14 天）

### 3.1 总体架构图

```
      ODIR 图像                           DDR 图像
         │                                   │
    随机增强×2                          预处理
    ┌────┴────┐                              │
    x⁽¹⁾     x⁽²⁾                           x
    │         │                              │
    └────┬────┘                              │
         │                                   │
    ┌────▼───────────────────────────────────▼────┐
    │        共享 RetFound ViT-Large               │
    │   ┌──────────────────────────────────────┐  │
    │   │  第 1-17 层 (冻结 70%，阶段一)        │  │
    │   │  第 18-24 层                         │  │
    │   └──────────────────────────────────────┘  │
    │   输出: [CLS] + Patch Tokens + 注意力图      │
    └──┬──────────┬──────────────┬────────────┬──┘
       │          │              │            │
  ┌────▼───┐ ┌───▼──────┐ ┌────▼────┐ ┌─────▼─────┐
  │  LPM   │ │ODIR Head │ │DDR Head │ │   CTAM    │
  │ L2一致 │ │MLP→Sigmoid│ │FPN→Seg  │ │Proj→XAttn │
  │ 注意力 │ │8 类多标签 │ │5 类分割 │ │对比对齐   │
  │ 增强   │ │          │ │         │ │           │
  └────┬───┘ └────┬─────┘ └────┬────┘ └─────┬─────┘
       │          │            │            │
       │     L_ODIR            │        投影到 256 维
       │   (ASL Loss)    L_DDR           │
       │               (Dice+CE)    ┌────▼────┐
       └──────────┬─────────────────┤ L_align │
                  │                 │ 对比损失 │
              L_lesion              └─────────┘
              (一致性)                  │
                  │                     │
       ┌──────────▼─────────────────────▼──────┐
       │          L_total = L_O + L_D           │
       │         + 0.1×L_lesion                 │
       │         + 0.05×L_align                 │
       └────────────────────────────────────────┘
```

### 3.2 各模块实现清单

#### 模块 A：共享编码器 (`src/models/encoder.py`)

```
功能：
- 加载 RetFound ViT-Large 预训练权重
- 位置编码插值（256→512）
- 多层级特征提取（第 12/18/24 层 Patch Tokens）
- [CLS] Token 输出（用于 ODIR 分类）
- 最后一层注意力图输出（用于 LPM）

关键类：
class RetFoundEncoder(nn.Module):
    def __init__(self, pretrained_path, freeze_ratio=0.7):
        ...
    def forward(self, x, return_attention=True, return_multilevel=True):
        # x: (B, 3, 512, 512)
        # returns: {
        #   'cls_token': (B, 1024),
        #   'patch_tokens_l12': (B, 1024, 1024),
        #   'patch_tokens_l18': (B, 1024, 1024),
        #   'patch_tokens_l24': (B, 1024, 1024),
        #   'attention_maps': (B, 16, 1025, 1025),  # 最后一层多头注意力
        # }
```

**注意事项**：
- RetFound 是基于 MAE 预训练的 ViT，需要确认其 forward 是否返回中间层特征
- 如果 HuggingFace 的 ViT 不返回注意力权重，需要设置 `output_attentions=True`
- 提取第 12/18/24 层特征可能需要在 forward 中手动截取 hidden_states

#### 模块 B：LPM 病灶感知模块 (`src/models/lpm.py`)

```
功能：
1. 注意力一致性约束：对同一图两次增强得到的注意力图做 L2 Loss
2. 空间注意力增强：用注意力图按元素乘编码器特征

关键类：
class LesionPerceptionModule(nn.Module):
    def __init__(self):
        ...
    def compute_consistency_loss(self, attn1, attn2):
        """L2 损失，约束两次增强的注意力分布一致"""
        return F.mse_loss(attn1, attn2)

    def enhance_features(self, features, attention_map):
        """用注意力图加权增强特征"""
        # attention_map: (B, N_patches, N_patches) → resize 到特征空间分辨率
        # features: (B, N_patches, D)
        # returns: enhanced_features (B, N_patches, D)
        ...

    def forward(self, features, attn1, attn2):
        loss_consistency = self.compute_consistency_loss(
            self._avg_heads(attn1), self._avg_heads(attn2)
        )
        attn_avg = (self._avg_heads(attn1) + self._avg_heads(attn2)) / 2
        enhanced = self.enhance_features(features, attn_avg)
        return enhanced, loss_consistency
```

**实现细节**：
- `_avg_heads`：取最后一层所有注意力头的均值（论文中 `Ā`）
- 注意力图 reshape：`(B, heads, N+1, N+1)` → 去掉 CLS → `(B, N, N)` → resize 到 `(B, H, W)`
- 增强操作为 Hadamard 积（逐元素乘）

#### 模块 C：CTAM 跨任务对齐模块 (`src/models/ctam.py`)

```
功能：
1. 共享空间投影：ODIR/DDR → 各自投影矩阵 → 256 维 → L2 归一化
2. 跨任务交叉注意力：ODIR (Q) × DDR (K,V)
3. 有监督对比对齐损失

关键类：
class CrossTaskAlignmentModule(nn.Module):
    def __init__(self, dim_in=1024, dim_proj=256):
        self.proj_O = nn.Linear(1024, 256)  # ODIR 投影（不共享）
        self.proj_D = nn.Linear(1024, 256)  # DDR 投影（不共享）
        self.cross_attn = nn.MultiheadAttention(256, num_heads=8, batch_first=True)
        self.alpha = 0.5  # 残差缩放系数

    def forward(self, F_ODIR, F_DDR, y_DR):
        """
        F_ODIR: (B_O, 1024)
        F_DDR:  (B_D, 1024)
        y_DR:   (B_O,)  ODIR 中每个样本是否为 DR 阳性
        """
        # 1. 投影 + L2 归一化
        Z_O = F.normalize(self.proj_O(F_ODIR), dim=-1)  # (B_O, 256)
        Z_D = F.normalize(self.proj_D(F_DDR), dim=-1)    # (B_D, 256)

        # 2. 交叉注意力增强（ODIR 查 DDR）
        Z_O_enhanced, _ = self.cross_attn(
            query=Z_O.unsqueeze(0),
            key=Z_D.unsqueeze(0),
            value=Z_D.unsqueeze(0)
        )
        Z_O_enhanced = Z_O + self.alpha * Z_O_enhanced.squeeze(0)

        # 3. 有监督对比对齐损失
        loss_align = self.supervised_contrastive_loss(Z_O, Z_D, y_DR)

        return Z_O_enhanced, Z_D, loss_align
```

**有监督对比对齐损失实现**（对应论文公式 20-21）：

```python
def supervised_contrastive_loss(self, Z_O, Z_D, y_DR, tau=0.07):
    """
    Z_O: (B_O, 256) — L2 归一化后的 ODIR 投影特征
    Z_D: (B_D, 256) — L2 归一化后的 DDR 投影特征
    y_DR: (B_O,) — DR 阳性标签（1=阳性, 0=阴性）
    tau: 温度系数

    正对：DR 阳性 ODIR ↔ DDR（语义一致）
    负对：DR 阴性 ODIR ↔ DDR（语义不一致）

    对应论文公式 (20)：对每个 DR 阳性 ODIR 样本，
    计算其与所有 DDR 样本的余弦相似度，拉近正样本、推远负样本。
    """
    B_O, B_D = Z_O.shape[0], Z_D.shape[0]
    pos_mask = (y_DR == 1)  # (B_O,)

    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=Z_O.device)  # 无 DR 阳性，跳过

    # 相似度矩阵：(B_O, B_D)
    sim = torch.matmul(Z_O, Z_D.T) / tau  # Z_O 和 Z_D 均已 L2 归一化

    # 对每个 DDR 样本 j：
    # 正样本：DR 阳性的 ODIR 样本
    # 负样本：所有 ODIR 样本（正负均包含）
    loss = 0.0
    for j in range(B_D):
        pos_sim = sim[pos_mask, j]                 # 正样本相似度
        all_sim = sim[:, j]                         # 所有 ODIR 样本相似度

        # InfoNCE：exp(pos_sim) / sum(exp(all_sim))
        numerator = torch.exp(pos_sim).sum()
        denominator = torch.exp(all_sim).sum()
        loss += -torch.log(numerator / denominator)

    return loss / B_D
```

**边界条件处理**：
- 当 batch 中无 DR 阳性样本（`pos_mask.sum() == 0`）时，跳过对齐损失计算 → 返回 0
- 在实际训练中通过加权重采样确保每 batch 至少 2 个 DR 阳性样本

#### 模块 D：预测头 (`src/models/heads.py`)

**ODIR 多标签分类头**：
```python
class ODIRClassifier(nn.Module):
    def __init__(self, dim_in=1024, num_classes=8):
        self.fc = nn.Linear(1024, 8)

    def forward(self, cls_token):
        return torch.sigmoid(self.fc(cls_token))  # (B, 8)
```

**DDR 病灶分割头**：
```python
class DDRSegmentator(nn.Module):
    """
    多层级预测架构：
    - 从第 12/18/24 层提取 Patch Token
    - 1×1 卷积对齐通道
    - 上采样至统一分辨率
    - 拼接 + 最终 1×1 卷积 → 5 类 softmax
    """
    def __init__(self, dims=[1024, 1024, 1024], num_classes=5, patch_size=16):
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(d, 256, 1) for d in dims
        ])
        self.final_conv = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 5, 1)
        )

    def forward(self, patch_tokens_l12, patch_tokens_l18, patch_tokens_l24, img_size=512):
        # patch_tokens: (B, N_patches, dim) → (B, dim, H, W)
        f12 = self._to_spatial(patch_tokens_l12, img_size)
        f18 = self._to_spatial(patch_tokens_l18, img_size)
        f24 = self._to_spatial(patch_tokens_l24, img_size)

        # 1×1 卷积对齐 + 上采样
        f12 = F.interpolate(self.lateral_convs[0](f12), scale_factor=4, mode='bilinear')
        f18 = F.interpolate(self.lateral_convs[1](f18), scale_factor=4, mode='bilinear')
        f24 = F.interpolate(self.lateral_convs[2](f24), scale_factor=4, mode='bilinear')

        fused = torch.cat([f12, f18, f24], dim=1)  # (B, 768, 128, 128)
        out = self.final_conv(fused)                 # (B, 5, 128, 128)
        out = F.interpolate(out, size=(img_size, img_size), mode='bilinear')
        return F.softmax(out, dim=1)
```

#### 模块 E：完整框架 (`src/models/retlesionuni.py`)

```python
class RetLesionUni(nn.Module):
    """
    组装所有模块的顶层模型
    """
    def __init__(self, pretrained_path, freeze_ratio=0.7):
        super().__init__()
        self.encoder = RetFoundEncoder(pretrained_path, freeze_ratio)
        self.lpm = LesionPerceptionModule()
        self.ctam = CrossTaskAlignmentModule()
        self.odir_head = ODIRClassifier()
        self.ddr_head = DDRSegmentator()

    def forward(self, x_odir, x_ddr, y_dr_labels=None, training=True):
        """
        x_odir: (B_O, 3, 512, 512) — ODIR 图像
        x_ddr:  (B_D, 3, 512, 512) — DDR 图像
        y_dr_labels: (B_O,) — ODIR 的 DR 标签（CTAM 需要）
        """
        results = {}

        # === ODIR 前向 ===
        # 两次随机增强 → 获取注意力图
        enc_out1 = self.encoder(x_odir_aug1, return_attention=True)
        enc_out2 = self.encoder(x_odir_aug2, return_attention=True)

        # LPM
        F_odir_enhanced, loss_consistency = self.lpm(
            enc_out1['patch_tokens_l24'],
            enc_out1['attention_maps'],
            enc_out2['attention_maps']
        )

        # ODIR 分类
        F_odir_cls = enc_out1['cls_token']  # 或使用增强后的全局特征
        odir_logits = self.odir_head(F_od_enhanced_pooled)

        # === DDR 前向 ===
        ddr_enc = self.encoder(x_ddr, return_multilevel=True)
        ddr_seg = self.ddr_head(
            ddr_enc['patch_tokens_l12'],
            ddr_enc['patch_tokens_l18'],
            ddr_enc['patch_tokens_l24']
        )
        F_ddr = ddr_enc['cls_token']

        # === CTAM ===
        Z_odir_enhanced, Z_ddr, loss_align = self.ctam(
            F_odir_cls, F_ddr, y_dr_labels
        )

        results['odir_logits'] = odir_logits
        results['ddr_seg'] = ddr_seg
        results['loss_consistency'] = loss_consistency
        results['loss_align'] = loss_align
        results['Z_odir'] = Z_odir_enhanced
        results['Z_ddr'] = Z_ddr
        return results
```

---

## 四、损失函数实现（预计 3-5 天）

### 4.1 Asymmetric Loss（ODIR 分类）

```python
# src/losses/asl_loss.py
class AsymmetricLoss(nn.Module):
    """
    论文公式 (7)：
    L_ODIR = -(1/N) Σ_i Σ_j [y_ij (1-p_ij)^γ⁺ log(p_ij) + (1-y_ij) p_ij^γ⁻ log(1-p_ij)]

    γ⁺ = 1, γ⁻ = 4
    """
    def __init__(self, gamma_pos=1.0, gamma_neg=4.0):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, pred, target):
        # pred: (B, 8), target: (B, 8)
        pos = target * ((1 - pred) ** self.gamma_pos) * torch.log(pred + 1e-8)
        neg = (1 - target) * (pred ** self.gamma_neg) * torch.log(1 - pred + 1e-8)
        return -(pos + neg).mean()
```

### 4.2 Dice + CE Loss（DDR 分割）

```python
# src/losses/dice_loss.py
class DiceCELoss(nn.Module):
    """
    论文公式 (9-11)：
    L_DDR = 0.7 * L_Dice + 0.3 * L_CE
    """
    def __init__(self, dice_weight=0.7, ce_weight=0.3):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        # pred: (B, 5, H, W), target: (B, H, W) — 类别索引
        loss_ce = self.ce(pred, target)

        # Dice loss（多类平均）
        pred_softmax = F.softmax(pred, dim=1)
        loss_dice = 0.0
        for c in range(pred.shape[1]):
            pred_c = pred_softmax[:, c, ...]
            target_c = (target == c).float()
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            loss_dice += 1 - (2 * intersection + 1) / (union + 1)
        loss_dice /= pred.shape[1]

        return self.dice_weight * loss_dice + self.ce_weight * loss_ce
```

### 4.3 注意力一致性 Loss（LPM）

```python
# src/losses/consistency_loss.py
def attention_consistency_loss(attn1, attn2):
    """
    论文公式 (15)：
    L_consistency = ||A(x⁽¹⁾) - A(x⁽²⁾)||₂²

    attn1, attn2: (B, heads, N+1, N+1)
    对多头取均值后计算 L2 距离
    """
    A1 = attn1.mean(dim=1)  # (B, N+1, N+1)
    A2 = attn2.mean(dim=1)  # (B, N+1, N+1)
    # 去掉 CLS Token 维度（仅保留 patch 之间的注意力）
    A1 = A1[:, 1:, 1:]  # (B, N, N)
    A2 = A2[:, 1:, 1:]  # (B, N, N)
    return F.mse_loss(A1, A2)
```

### 4.4 总体损失组合

```python
# 在 trainer.py 中
lambda_lesion = 0.1
lambda_align = 0.05

loss_total = (loss_odir + loss_ddr
              + lambda_lesion * loss_consistency
              + lambda_align * loss_align)
```

---

## 五、训练流程（预计 5-7 天调试）

### 5.1 两阶段渐进式微调

```python
# src/train/trainer.py（伪代码）
class RetLesionUniTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def train(self):
        # ===== 第一阶段：冻结前 70%，训练 30 epochs =====
        self._freeze_encoder(ratio=0.7)
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-3, weight_decay=1e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=30)

        for epoch in range(30):
            for batch_odir, batch_ddr in dataloader:
                # 仅使用主任务损失（不激活 LPM 和 CTAM）
                loss = loss_odir + loss_ddr
                loss.backward()
                optimizer.step()

        # ===== 第二阶段：全参数，训练 70 epochs =====
        self._unfreeze_all()
        optimizer = AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=70)

        for epoch in range(70):
            for batch_odir, batch_ddr in dataloader:
                # 全损失（LPM + CTAM 激活）
                # 前 10 个 epoch 线性升温 L_lesion 和 L_align
                warmup_ratio = min(1.0, epoch / 10)
                results = self.model(batch_odir, batch_ddr, training=True)
                loss = (loss_odir + loss_ddr
                        + warmup_ratio * lambda_lesion * results['loss_consistency']
                        + warmup_ratio * lambda_align * results['loss_align'])
                loss.backward()
                optimizer.step()

    def _freeze_encoder(self, ratio=0.7):
        total_layers = 24  # ViT-Large
        freeze_until = int(total_layers * ratio)  # 层 0-16（17 层冻结）
        for i, block in enumerate(self.model.encoder.blocks):
            if i < freeze_until:
                for param in block.parameters():
                    param.requires_grad = False
```

### 5.2 Batch 构造策略

```python
# ODIR 和 DDR 各自独立采样
class JointDataLoader:
    def __iter__(self):
        odir_iter = iter(self.odir_loader)  # B_O=8, 加权采样确保至少 2 个 DR 阳性
        ddr_iter = iter(self.ddr_loader)    # B_D=8

        while True:
            batch_odir, y_dr = next(odir_iter)
            batch_ddr, masks = next(ddr_iter)

            # 对 ODIR 施加两次随机增强（LPM 需要）
            odir_aug1 = self.augment(batch_odir)
            odir_aug2 = self.augment(batch_odir)

            yield {
                'odir_aug1': odir_aug1,
                'odir_aug2': odir_aug2,
                'odir_labels': batch_odir.labels,
                'y_dr': y_dr,
                'ddr_images': batch_ddr,
                'ddr_masks': masks,
            }
```

**确保 DR 阳性采样的策略**：
```python
class WeightedODIRSampler(Sampler):
    def __init__(self, dataset):
        # DR 阳性样本权重 ×3
        weights = [3.0 if y['dr'] == 1 else 1.0 for y in dataset.labels]
        self.weights = weights

    def __iter__(self):
        return iter(torch.multinomial(
            torch.tensor(self.weights),
            num_samples=len(self.weights),
            replacement=True
        ))
```

### 5.3 显存管理

```
配置：
- Batch size: B_O=8, B_D=8, 合计 16 张 512×512 RGB
- RetFound ViT-Large: ~300M 参数
- FP32 训练：~1.2GB 参数 + ~18GB 中间激活 ≈ 20-22GB
- RTX 4090 24GB 刚好够用

如果显存不足的备选方案：
1. 使用 gradient checkpointing（torch.utils.checkpoint）
2. 减小 batch size 至 B_O=6, B_D=6
3. 使用混合精度训练（FP16 → 约省 40% 显存）
```

---

## 六、实验清单与执行顺序

### 实验 0：预实验验证（必须最先做）

**目标**：验证代码和训练管线能跑通

- [ ] 用 ResNet-50 在 ODIR 上训练 5 个 epoch，验证数据加载和分类头
- [ ] 用 U-Net 在 DDR 上训练 5 个 epoch，验证数据加载和分割头
- [ ] 加载 RetFound 权重，前向传播一次 512×512 图像，确认显存占用
- [ ] 跑通一次完整的 RetLesionUni 训练迭代（1 个 batch）

### 实验 1：单任务基线（预计 2-3 天）

| 方法 | 数据集 | GPU 时间 | 备注 |
|------|--------|----------|------|
| ResNet-50 | ODIR | ~4h | ImageNet 预训练 |
| U-Net | DDR | ~4h | 从头训练 |
| RetFound-Single (分类) | ODIR | ~8h | 单任务微调 |
| RetFound-Single (分割) | DDR | ~8h | 单任务微调 |
| TransUNet | DDR | ~6h | Transformer+CNN |

**产出**：
- `experiments/exp01_baseline/resnet50_odir/`
- `experiments/exp01_baseline/unet_ddr/`
- `experiments/exp01_baseline/retfound_single/`
- 主结果表第 1-4 行的实验数据（当前占位值待替换）

### 实验 2：多任务基线（预计 3-4 天）

| 方法 | GPU 时间 | 备注 |
|------|----------|------|
| MTAN | ~15h | 需要实现软注意力机制 |
| PAD-Net | ~15h | 需要实现预测蒸馏 |
| MMoE | ~15h | 需要实现多门控专家 |

**注意**：这些方法使用其原始 backbone（ResNet），不要替换为 RetFound。

**产出**：主结果表第 5-7 行的实验数据

### 实验 3：RetLesionUni 完整训练（预计 4-5 天）

**这是核心实验，需要跑 100 epochs（~100 小时）**

```
阶段一（30 epochs）：lr=1e-3, 冻结前 70%, 仅 L_ODIR + L_DDR
阶段二（70 epochs）：lr=1e-4, 全参数, L_total（含 LPM + CTAM 渐进激活）

运行命令：
python src/train/trainer.py \
    --exp_name retlesionuni_full \
    --batch_size_odir 8 \
    --batch_size_ddr 8 \
    --epochs_stage1 30 \
    --epochs_stage2 70 \
    --lr_stage1 1e-3 \
    --lr_stage2 1e-4 \
    --lambda_lesion 0.1 \
    --lambda_align 0.05 \
    --warmup_epochs 10
```

**产出**：主结果表 RetLesionUni 行的实验数据

### 实验 4：消融实验（预计 6-8 天）

| 实验组 | 说明 | GPU 时间 |
|--------|------|----------|
| Baseline（仅 L_O+L_D） | 无 LPM，无 CTAM | ~80h |
| +LPM（仅病灶感知） | Baseline + LPM | ~85h |
| +CTAM（仅跨任务对齐） | Baseline + CTAM | ~90h |
| Full Model | LPM + CTAM | ~100h |

**产出**：消融实验表的实验数据（5 组 × 5 次种子 = 25 次训练运行，主消融可先跑 1 次种子）

### 实验 5：损失权重分析（预计 5-7 天）

网格搜索 λ_lesion ∈ {0.0, 0.05, 0.1, 0.15, 0.2}, λ_align ∈ {0.0, 0.05, 0.1}

**产出**：权重分析表的实验数据

### 实验 6：多随机种子稳定性（与实验 3 并行）

对 Full Model 用 5 个不同随机种子（42, 123, 456, 789, 1024）各训练一次。

**产出**：一致性表的实验数据

### 实验 7：RetFound + 多任务基线（预计 4-5 天）

| 方法 | GPU 时间 |
|------|----------|
| RetFound + MTAN | ~90h |
| RetFound + MMoE | ~90h |
| DiffDGSSv2 | ~30h（仅 DDR 分割，从 GitHub 复现） |

**产出**：baselines 实验数据（论文中当前数值为占位符，需全部由实验生成）

### 实验 8：可解释性分析（预计 2-3 天，在实验 3 产出模型后执行）

- [ ] Attention Rollout 生成热力图（`src/eval/explainability.py`）
- [ ] 注意力-病灶重叠率定量分析（Precision/Recall/IoU）
- [ ] 病灶贡献度消融（逐个掩码注意力区域）
- [ ] Per-lesion 细粒度分割分析
- [ ] 失败案例收集与分析

### 实验 9：跨域泛化（预计 2-3 天，P3 优先级）

- [ ] IDRiD 数据集预处理
- [ ] DDR 训练模型在 IDRiD 上零样本推理
- [ ] 计算跨域性能指标

---

## 七、评估指标实现

### 7.1 ODIR 分类指标

```python
# src/eval/metrics.py
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def compute_odir_metrics(preds, targets):
    """
    preds: (N, 8) sigmoid 概率
    targets: (N, 8) 多标签
    论文定义：Accuracy = 所有标签精确匹配的比例
    """
    # 阈值化（>0.5 为正类）
    pred_binary = (preds > 0.5).astype(int)

    # Exact Match Accuracy：所有 8 个标签全对才算对
    acc = (pred_binary == targets).all(axis=1).mean()

    # Macro F1：逐类别计算 F1 后取平均
    f1 = f1_score(targets, pred_binary, average='macro')

    # AUC-ROC：逐类别计算后取宏平均
    auc = roc_auc_score(targets, preds, average='macro')

    return {'accuracy': acc, 'f1': f1, 'auc': auc}
```

### 7.2 DDR 分割指标

```python
def compute_ddr_metrics(pred_mask, gt_mask, num_classes=5):
    """
    pred_mask: (H, W) 预测类别索引
    gt_mask: (H, W) 真实类别索引
    """
    iou_per_class = []
    dice_per_class = []

    for c in range(num_classes):
        pred_c = (pred_mask == c)
        gt_c = (gt_mask == c)
        intersection = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()

        iou = intersection / (union + 1e-8)
        dice = 2 * intersection / (pred_c.sum() + gt_c.sum() + 1e-8)

        iou_per_class.append(iou)
        dice_per_class.append(dice)

    mIoU = np.mean(iou_per_class)
    mDice = np.mean(dice_per_class)

    return {'mIoU': mIoU, 'mDice': mDice,
            'per_class_iou': iou_per_class,
            'per_class_dice': dice_per_class}
```

### 7.3 病灶定位 IoU（A_loc）

```python
def compute_lesion_localization(attention_map, gt_mask, threshold=0.3):
    """
    论文公式 (22)：A_loc = |Binary(A) ∩ M| / |Binary(A) ∪ M|
    attention_map: (H, W) Attention Rollout 生成的注意力图
    gt_mask: (H, W) 病灶区域掩码（4 类病灶的并集）
    """
    attn_binary = (attention_map > threshold).astype(int)
    intersection = (attn_binary & gt_mask).sum()
    union = (attn_binary | gt_mask).sum()
    return intersection / (union + 1e-8)
```

### 7.4 Attention Rollout 实现

```python
# src/eval/explainability.py
def attention_rollout(attn_matrices, discard_ratio=0.0, head_fusion='mean'):
    """
    Abnar & Zuidema (ACL 2020)
    attn_matrices: list of (heads, N+1, N+1) — 所有层的注意力矩阵
    通过累乘各层注意力矩阵得到累积注意力分布
    """
    # 多头融合
    if head_fusion == 'mean':
        attn = torch.stack([a.mean(dim=0) for a in attn_matrices])
    # 去掉 CLS Token
    attn = attn[:, 1:, 1:]  # (L, N, N)
    # 加入残差连接（单位矩阵）
    eye = torch.eye(attn.shape[-1], device=attn.device)
    attn = [0.5 * a + 0.5 * eye for a in attn]
    # 累乘
    rollout = attn[0]
    for a in attn[1:]:
        rollout = torch.matmul(a, rollout)
    return rollout  # (N, N)
```

---

## 八、统计检验

### 配对 t 检验

```python
# src/eval/statistical_test.py
from scipy.stats import ttest_rel

def paired_t_test(results_ours, results_baseline, metrics):
    """
    results_ours: (5, n_metrics) — 5 次随机种子的结果
    results_baseline: (5, n_metrics)
    """
    p_values = {}
    for i, metric in enumerate(metrics):
        t_stat, p_val = ttest_rel(results_ours[:, i], results_baseline[:, i])
        p_values[metric] = p_val
    return p_values
```

---

## 九、执行时间线

```
Week 1-2:  环境搭建 + 数据准备 + 预实验验证
Week 3-4:   模型实现（encoder + LPM + CTAM + heads + losses）
Week 5:     训练管线调试 + 单任务基线实验（实验 1）
Week 6:     多任务基线实验（实验 2）
Week 7-8:   RetLesionUni 完整训练（实验 3 + 6，5 次种子可并行）
Week 9:     消融实验（实验 4）
Week 10:    权重分析 + RetFound 多任务基线（实验 5 + 7）
Week 11:    可解释性分析 + 跨域泛化（实验 8 + 9）
Week 12:    图表制作 + 论文最终修改
```

**总计**：约 12 周（3 个月），利用单张 RTX 4090。

**GPU 时间汇总**：
- 所有 5 次种子消融实验：~500 GPU-hours
- 所有基线方法：~200 GPU-hours
- 权重分析：~150 GPU-hours
- 总计：~850 GPU-hours ≈ 35 天（在单张 4090 上可用 3 个月完成）

---

## 十、关键风险与缓解措施

| 风险 | 概率 | 缓解 |
|------|------|------|
| RetFound 预训练权重不兼容 | 中 | 提前下载并测试；备选方案：使用 ViT-Large ImageNet 预训练 + 领域适配 |
| 24GB 显存不够 | 中 | 混合精度训练（FP16）、gradient checkpointing、减小 batch size 至 6 |
| ODIR/DDR 数据集无法获取 | 低 | 这两个是公开数据集，但需要确认下载链接仍有效 |
| 100 小时训练中途崩溃 | 中 | 每隔 10 epoch 保存 checkpoint；使用 `torch.cuda.amp` 提高稳定性 |
| 消融实验效果不明显 | 中 | 如果某个模块的提升太小，先检查实现是否正确，再调整损失权重 |
| DiffDGSSv2 复现困难 | 高 | 作者有开源代码（github.com/Xyporz/DiffDGSSv2），优先从 GitHub 复现 |

---

## 十一、待补充项清单

在正式开始编码前，需要确认以下信息：

- [ ] ODIR 数据集下载链接和标签格式
- [ ] DDR 数据集下载链接和标注格式
- [ ] RetFound ViT-Large 预训练权重的下载和加载方式
- [ ] 导师是否同意论文中所述的所有实验设计
- [ ] 目标期刊的具体图表和参考文献格式要求
- [ ] 是否需要伦理审批声明（DR 筛查 AI 涉及患者数据）

---

> **最后提醒**：论文手稿中所有表格的所有数值均为占位符，不存在"真实数据"与"模拟数据"的区分——每一个数字都需要通过实验产生。当前草稿中已用 Python 确认过全部占位数值的跨表数学一致性（mDice、IoU、A_loc 等），但占位符本身不来自任何实验。实验完成后，必须把手稿中所有数值替换为实测结果，并重新验证跨表一致性。
