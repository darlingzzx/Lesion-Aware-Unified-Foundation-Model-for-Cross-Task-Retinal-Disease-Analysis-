# RetLesionUni 训练操作指南

> 给新 Claude 会话读的完整操作文档。读完即可独立完成环境搭建、数据预处理、模型训练和评估。

---

## 一、项目概述

**论文**: Lesion-Aware Unified Foundation Model for Cross-Task Retinal Disease Analysis

**任务**: 共享 RetFound ViT-Large 编码器，同时做两件事：
- **ODIR 数据集** — 8 标签多标签眼底疾病分类 (Normal/Diabetes/Glaucoma/Cataract/AMD/Hypertension/Myopia/Other)
- **DDR 数据集** — 4 类像素级病灶分割 (EX硬性渗出/HE出血/MA微动脉瘤/SE软性渗出 + 背景)

**核心创新模块**:
1. **LPM** (Lesion Perception Module): 同一图像两次增强 → 注意力图一致性约束 → 注意力加权增强特征
2. **CTAM** (Cross-Task Alignment Module): 各自投影到256维 → 交叉注意力 → 以DR标签做监督对比对齐

**训练策略**: 两阶段微调
| | 阶段一 (1-30 epoch) | 阶段二 (31-100 epoch) |
|---|---|---|
| 冻结 ViT 层 | 前 70% (层 0-16) | 0% |
| LR | 1e-3 | 1e-4 |
| LPM / CTAM | 关闭 | 启用 (前10 epoch 线性升温) |
| 损失 | L_O + L_D | L_O + L_D + 0.1×L_lesion + 0.05×L_align |

**硬件要求**: RTX 4090 24GB × 1，训练约 100 小时。需要 ≥ 64GB RAM，≥ 200GB 磁盘。

---

## 二、环境搭建

### 2.1 创建 Conda 环境

```bash
conda create -n retlesionuni python=3.10 -y
conda activate retlesionuni
```

### 2.2 安装 PyTorch (CUDA 12.1)

```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

### 2.3 安装项目及依赖

在项目根目录执行：

```bash
pip install -e .
```

这会安装 `pyproject.toml` 中声明的所有依赖：
- `transformers>=4.30.0`, `timm>=0.9.0` — ViT 模型
- `albumentations>=1.3.0` — 数据增强
- `opencv-python>=4.8.0` — TIFF 读取（PIL 无法读取 PackBits 压缩 TIFF）
- `monai>=1.2.0` — 医学图像工具
- `scikit-learn`, `scipy`, `pandas`, `matplotlib`, `seaborn` — 评估和可视化
- `tensorboard`, `tqdm`, `einops` — 训练工具
- `pyyaml`, `openpyxl` — 配置和数据解析

### 2.4 下载 RetFound 预训练权重

**重要**：必须使用在 160 万张视网膜图像上预训练的 RetFound 权重，不能用 ImageNet 权重替代。

**下载步骤**：

1. 访问 https://huggingface.co/open-eye/RETFound_MAE
2. 找到 "Colour fundus image" 行，点击 `download` 链接
   - Google Drive 文件 ID: `1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE`
   - 文件名: `RETFound_cfp_weights.pth`（约 2 GB）
3. 放到项目 `pretrained/` 目录下：

```bash
mkdir -p pretrained
# 将下载的 RETFound_cfp_weights.pth 放到 pretrained/ 目录
```

**备选下载方式** (如果 Google Drive 被墙):
```bash
# 方式1: 用 gdown
pip install gdown
gdown "https://drive.google.com/uc?id=1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE" -O pretrained/RETFound_cfp_weights.pth

# 方式2: 用 wget 配合 Google Drive 直链
# 需要先获取确认 token，或用浏览器下载
```

**验证**：
```bash
python -c "
import torch
ckpt = torch.load('pretrained/RETFound_cfp_weights.pth', map_location='cpu', weights_only=False)
print('Keys:', list(ckpt.keys()))
print('Model keys:', len(ckpt['model']))
# 应该输出: Keys: ['model', 'epoch', ...] 且 model keys ~300+
"
```

### 2.5 确认 GPU 可用

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## 三、数据准备

### 3.1 数据目录结构 (已有)

```
data/
├── label_mapping.json                    # 类别映射 (ODIR←→DDR)
├── DDR dataset/
│   ├── lesion_detection/
│   │   ├── train/                        # 383 个 XML (Pascal VOC, 有重复 bbox)
│   │   ├── valid/                        # 149 个 XML
│   │   └── test/                         # 225 个 XML
│   └── lesion_segmentation/
│       ├── train/image/                  # 383 张 JPG + label/{EX,HE,MA,SE}/
│       ├── valid/image/                  # 149 张 + "segmentation label"/{EX,HE,MA,SE}/  ← 注意空格!
│       └── test/image/                   # 225 张 + label/{EX,HE,MA,SE}/
└── ODIR/
    ├── train/                            # 4900 张 JPG + train_annotations.xlsx
    ├── valid/                            # 1050 张 JPG + valid_annotations.xlsx
    └── test/                             # 1050 张 JPG + test_annotations.xlsx
```

### 3.2 数据标注格式

**DDR 病灶检测 XML** (Pascal VOC):
```xml
<object>
    <name>ma</name>           <!-- ma/he/ex/se (小写) -->
    <bndbox>
        <xmin>677</xmin><ymin>857</ymin><xmax>683</xmax><ymax>865</ymax>
    </bndbox>
</object>
```
注意：**所有 train XML 都有重复 bbox（每个出现 2-3 次）**，预处理会自动去重。

**DDR 分割标注**: 每种病灶类型一个 TIFF 文件，单通道 uint8 (0=背景, 255=病灶)。TIFF 使用 PackBits 压缩，必须用 OpenCV 读取。

**ODIR 标注**: 每个 Excel 文件有 15 列：
- `ID`, `Left-Fundus`, `Right-Fundus` — 患者 ID 和左右眼文件名
- `N, D, G, C, A, H, M, O` — 8 个二值标签 (Normal/Diabetes/Glaucoma/Cataract/AMD/Hypertension/Myopia/Other)
- 标签是**患者级别**的（双眼共享），预处理展开为图像级别

**类别映射** (`label_mapping.json`):
- DDR 的所有 DR 分级 (No DR/Mild/Moderate/Severe/Proliferative) → ODIR Diabetes
- 用于 CTAM 的 DR 阳性/阴性判定

### 3.3 数据质量问题 (预处理已处理)

| 问题 | 处理方式 |
|------|----------|
| DDR XML bbox 重复 (最多 3 次) | `set()` 去重 |
| DDR valid 目录名为 "segmentation label" (含空格) | 自动检测目录名 |
| TIFF 用 PackBits 压缩, PIL 无法读像素 | 用 `cv2.imread()` |
| ODIR 标签是患者级别 | 展开为图像级 |
| 图像分辨率不统一 (~1956~3888) | 训练时 resize 到 512×512 |
| ODIR 类别不均衡 (Hypertension 3%) | Asymmetric Loss (γ⁺=1, γ⁻=4) 处理 |

### 3.4 运行预处理

```bash
# 首次运行 (生成 outputs/preprocessed/)
python scripts/preprocess.py

# 强制重新生成
python scripts/preprocess.py --force

# 使用自定义配置
python scripts/preprocess.py --config configs/default.yaml
```

预处理输出 (`outputs/preprocessed/`):
```
ddr/
├── train/  {images/, masks/{id}.npy, bboxes/{id}_bboxes.json, metadata.csv}
├── valid/  ...
└── test/   ...
odir/
├── train/  {images/, metadata.csv}
├── valid/  ...
└── test/   ...
```

验证预处理结果:
```bash
python -c "
import pandas as pd
print('DDR train:', len(pd.read_csv('outputs/preprocessed/ddr/train/metadata.csv')))  # 383
print('ODIR train:', len(pd.read_csv('outputs/preprocessed/odir/train/metadata.csv'))) # 4900
"
```

---

## 四、本地验证 (CPU 测试)

在正式训练前，用 test config 跑一遍确保代码正确：

### 4.1 运行所有测试

```bash
# 损失函数测试
KMP_DUPLICATE_LIB_OK=TRUE python tests/test_losses.py

# 模型前向测试
KMP_DUPLICATE_LIB_OK=TRUE python tests/test_model_forward.py

# 集成测试 (数据加载 + 训练步骤)
KMP_DUPLICATE_LIB_OK=TRUE python tests/test_training_step.py
```

> Windows 用户需要 `KMP_DUPLICATE_LIB_OK=TRUE` 解决 OpenMP DLL 冲突。

### 4.2 快速训练测试 (CPU, 1 epoch)

```bash
python scripts/train.py --config configs/test.yaml
```

test config 使用 ViT-Tiny + 128px + batch_size=2，1 个 epoch 应该几分钟内完成。

---

## 五、正式训练

### 5.1 修改配置

编辑 `configs/default.yaml` 确认以下关键参数：

```yaml
model:
  pretrained: true                              # 设为 true
  pretrained_path: "pretrained/retfound.pth"    # RetFound 权重路径
  img_size: 512
  backbone: "vit_large_patch16_224"
  # ... 其他保持默认

training:
  batch_size_odir: 8
  batch_size_ddr: 8
  num_workers: 4                                # 根据 CPU 核心数调整
  # ... 其他保持默认
```

### 5.2 启动训练

```bash
python scripts/train.py
# 或指定实验名
python scripts/train.py --overrides exp_name=my_experiment
```

### 5.3 监控训练

```bash
tensorboard --logdir outputs/logs
```

### 5.4 训练过程

- **阶段一** (30 epochs): 冻结前 17 层 ViT，仅训练分类头和分割头。LR=1e-3。
- **阶段二** (70 epochs): 解冻全部参数，启用 LPM + CTAM（前 10 epoch 线性升温）。LR=1e-4。
- Checkpoint 保存在 `outputs/checkpoints/{exp_name}/`
- 每 10 个 epoch 保存一次，最佳模型保存为 `best_model.pth`

### 5.5 显存优化 (如果 24GB 不够)

```bash
# 方法1: 减小 batch size
python scripts/train.py --overrides training.batch_size_odir=6 training.batch_size_ddr=6

# 方法2: 混合精度 (修改 default.yaml 或覆盖)
# training.mixed_precision: "fp16"

# 方法3: Gradient checkpointing
# training.gradient_checkpointing: true
```

### 5.6 从 checkpoint 恢复训练

修改 `src/retlesionuni/train/trainer.py` 或在训练循环中添加 `load_checkpoint()` 调用（当前 trainer 不支持断点续训，建议一次跑完）。

---

## 六、评估

### 6.1 评估测试集

```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint outputs/checkpoints/retlesionuni_full/best_model.pth
```

输出:
- ODIR: Accuracy, Macro F1, Macro AUC
- DDR: mIoU, mDice (包括背景和仅病灶)
- Per-class Dice/IoU (BG/EX/HE/MA/SE)

### 6.2 关键指标目标 (论文预期值)

| 指标 | 目标值 |
|------|--------|
| ODIR Accuracy | ~87.3% |
| ODIR Macro F1 | ~0.865 |
| ODIR AUC | ~0.935 |
| DDR mIoU | ~75.6% |
| DDR mDice | ~0.826 |
| A_loc (病灶定位) | ~0.742 |

---

## 七、代码架构说明

### 7.1 数据流

```
原始数据 (data/)
  → scripts/preprocess.py (去重/合并/展开)
  → outputs/preprocessed/ (.npy + .json + .csv)
  → DDRDataset / ODIRDataset (torch Dataset)
  → JointDataLoader (ODIR WeightedRandomSampler)
  → scripts/train.py → RetLesionUni → checkpoints
```

### 7.2 模型架构

```
ODIR Image (aug view 1 & 2)              DDR Image
        │                                      │
        ▼                                      ▼
   RetFound ViT-Large (共享权重)  ←───────────┘
        │
   ┌────┴────────────┬──────────────┐
   ▼                 ▼              ▼
  LPM           ODIR Head      DDR Head (FPN)
  (注意力一致性)  (Linear+Sigmoid)  (多层级→5类)
   │                 │              │
   ▼                 ▼              ▼
  L_lesion        L_ODIR         L_DDR
   │                 │              │
   └─────────┬───────┴──────────────┘
             │
        ┌────▼────┐
        │  CTAM   │
        │ 投影+交叉注意力+对比损失
        └────┬────┘
             ▼
         L_align
```

### 7.3 关键模块

| 文件 | 功能 |
|------|------|
| `models/encoder.py` | ViT 封装: pos_embed 双线性插值 (256→512), 冻结/解冻, 多层特征提取 (层12/18/24), 注意力图输出 |
| `models/lpm.py` | 多头注意力取均值 → 去 CLS token → 两次增强的注意力 MSE → 注意力加权特征 |
| `models/ctam.py` | 两个独立投影矩阵 (1024→256, 不共享) → L2 归一化 → Cross-Attention (ODIR query DDR) → 监督对比损失 |
| `models/heads.py` | ODIR: Linear(1024→8)+Sigmoid; DDR: 3层 FPN (1×1 Conv 对齐 → 上采样 → Concat → 2层 Conv) |
| `losses/asl_loss.py` | Asymmetric Loss: 正样本 (1-p)^1, 负样本 p^4 |
| `losses/dice_loss.py` | 0.7×Dice + 0.3×CE, 多类别 Dice 取均值 |
| `losses/contrastive_loss.py` | 对每个 DDR 样本: InfoNCE over all ODIR samples, DR阳性=正对 |

### 7.4 配置系统

- 所有参数在 `configs/default.yaml` 中
- 路径自动解析为绝对路径（相对于项目根目录）
- 支持 CLI 覆盖: `--overrides key.subkey=value`
- 路径以 `_root`, `_dir`, `_path`, `_cache` 结尾的会自动解析

---

## 八、实验清单

按照 `implementation-plan.md` 中的实验计划：

| 实验 | 说明 | 预计 GPU 时间 |
|------|------|---------------|
| 单任务基线 | ResNet-50/U-Net/RetFound-Single/TransUNet | ~30h |
| 多任务基线 | MTAN/PAD-Net/MMoE | ~45h |
| RetLesionUni 完整 | 核心实验，5 次种子 | ~500h |
| 消融实验 | Baseline/+LPM/+CTAM/Full | ~350h |
| 权重分析 | λ_lesion × λ_align 网格搜索 | ~150h |
| RetFound+基线 | RetFound+MTAN/MMoE + DiffDGSSv2 | ~210h |
| 可解释性 | Attention Rollout + A_loc | ~30h |
| 跨域泛化 | IDRiD 零样本测试 | ~20h |

**总计**: ~850 GPU-hours ≈ 35 天 (单卡 RTX 4090)

消融实验的 Baseline 组 (仅 L_O+L_D, 无 LPM/CTAM) 预期:
- Baseline mIoU < RetFound-Single mIoU → **确认负迁移**
- Full mIoU > RetFound-Single mIoU → **确认 LPM+CTAM 克服负迁移**

---

## 九、常见问题

### Q: 预处理报 "directories not found"
确认 `data/DDR dataset/` 和 `data/ODIR/` 目录存在，且内部结构与第三节一致。

### Q: TIFF 读取崩溃 (exit 127 或无输出)
PIL 无法读取 PackBits 压缩 TIFF。预处理已改用 `cv2.imread()`。确保 opencv-python 已安装。

### Q: OpenMP DLL 冲突 (libiomp5md.dll)
Windows 特定问题。设置环境变量: `KMP_DUPLICATE_LIB_OK=TRUE`

### Q: 显存不足 (CUDA Out of Memory)
1. 减小 batch size: `--overrides training.batch_size_odir=4 training.batch_size_ddr=4`
2. 启用混合精度: 编辑 config 设置 `training.mixed_precision: "fp16"`
3. 启用 gradient checkpointing: `training.gradient_checkpointing: true`

### Q: ODIR valid_annotations.xlsx 不存在
原始 ODIR 数据中 valid 和 test 目录需要有对应的 xlsx 文件。如果没有，从 train 目录复制一份并修改路径以测试代码，实际指标需要正确的标注。

### Q: 如何只跑消融实验的 Baseline 组
```bash
python scripts/train.py --overrides \
    model.lpm.enabled=false \
    model.ctam.enabled=false \
    training.stage2.use_lpm=false \
    training.stage2.use_ctam=false \
    exp_name=ablation_baseline
```

---

## 十、新增机器 Checklist

到达新机器后按顺序执行：

- [ ] 1. 克隆/复制整个项目目录
- [ ] 2. 确认数据在 `data/` 下
- [ ] 3. `conda create -n retlesionuni python=3.10 -y && conda activate retlesionuni`
- [ ] 4. 安装 PyTorch CUDA 版本
- [ ] 5. `pip install -e .` 安装项目依赖
- [ ] 6. 下载 RetFound 权重到 `pretrained/retfound.pth`
- [ ] 7. `python scripts/preprocess.py` 预处理数据
- [ ] 8. 修改 `configs/default.yaml` 中 `pretrained: true` 和正确路径
- [ ] 9. `python tests/test_training_step.py` 验证 (CPU 可用)
- [ ] 10. `python scripts/train.py` 开始训练
- [ ] 11. `tensorboard --logdir outputs/logs` 监控
- [ ] 12. 训练完成后 `python scripts/evaluate.py --checkpoint outputs/checkpoints/.../best_model.pth`
