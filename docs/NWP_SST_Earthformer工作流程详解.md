# Earthformer 西北太平洋 SSTA 预测 — 工作流程详解

> 基于 `scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py` 和 `cfg_nwp.yaml` 配置

---

## 目录

1. [任务定义](#1-任务定义)
2. [数据管线](#2-数据管线)
3. [模型架构总览](#3-模型架构总览)
4. [初始卷积编码 (InitialEncoder)](#4-初始卷积编码)
5. [位置编码 (PosEmbed)](#5-位置编码)
6. [编码器 — 3层层次结构](#6-编码器)
7. [Cuboid Attention 逐层策略](#7-cuboid-attention)
8. [全局向量 (Global Vectors)](#8-全局向量)
9. [解码器](#9-解码器)
10. [最终上采样 + 投影](#10-最终上采样)
11. [损失函数](#11-损失函数)
12. [评估指标](#12-评估指标)
13. [优化器与学习率调度](#13-优化器与学习率调度)
14. [完整数据形状流转表](#14-数据形状流转表)
15. [配置说明](#15-配置说明)
16. [训练流程](#16-训练流程)

---

## 1. 任务定义

| 项目 | 值 |
|------|-----|
| 输入 | 14 天 × 161×241 × 3 通道 [ssta, u10, v10] |
| 输出 | 3 天 × 161×241 × 1 通道 [ssta] |
| 区域 | 10°N–50°N, 120°E–180°E |
| 分辨率 | 0.25° |
| 数据源 | ERA5 2001-2022 日数据 |
| 模型参数量 | 18.4 M |

**与原始 ENSO 任务的关键区别**：

| | ENSO (原) | NWP SSTA (当前) |
|------|----------|-----|
| 网格 | 24×48 | 161×241 (34×更大) |
| 输入长度 | 12 月 | 14 天 |
| 输出长度 | 26 月 | 3 天 |
| 输入通道 | 4 (sst, t300, ua, va) | 3 (ssta, u10, v10) |
| 输出通道 | 4 | 1 (仅 ssta) |
| 编码器层数 | 2 层 | 3 层 |
| 注意力策略 | 全 axial | 逐层不同 |
| 全局向量 | 关闭 | 开启 (8 个) |
| 初始下采样 | [1,1,2] | [1,4,4] |
| 评估 | Niño 相关系数 | 逐像素 SSTA MSE/MAE/RMSE (°C) |

---

## 2. 数据管线

### 2.1 数据来源

ERA5 再分析数据，经过以下预处理步骤（各脚本独立运行）：

| 步骤 | 脚本 | 输入 → 输出 |
|------|------|-----------|
| 空间裁剪+单位转换 | `preprocess_nwp.py` | SST.nc + wind01-22.nc → SST_cropped.nc + Wind_cropped.nc |
| 海陆掩码 | `generate_ocean_mask.py` | SST_cropped.nc → mask.npy |
| 逐日气候态 | `compute_climatology.py` | SST_cropped.nc → climatology.nc |
| SSTA 计算 | `compute_ssta.py` | SST_cropped.nc + climatology.nc → ssta.nc |

### 2.2 训练时数据加载

**文件**: `src/earthformer/datasets/nw_pacific_dataset.py`

**函数**: `build_dataloaders(data_dir, batch_size, num_workers)`

**处理流程**：

```
1. 读取 ssta.nc + Wind_cropped.nc + mask.npy
       ↓
2. 归一化 (仅用训练集 2001-2020 计算 mean/std)
   ssta = (ssta - mean) / std
   u10  = (u10  - mean) / std
   v10  = (v10  - mean) / std
       ↓
3. 掩码陆地 (归一化后) → 陆地值 ≡ 0
   ssta = np.where(mask, ssta, 0.0)
   u10  = np.where(mask, u10,  0.0)
   v10  = np.where(mask, v10,  0.0)
       ↓
4. 堆叠通道
   data = stack([ssta, u10, v10], axis=-1)  → (8035, 161, 241, 3)
       ↓
5. 按年切分
   训练: 2001-2020 (7305天 → 7289样本)
   验证: 2021 (365天 → 349样本)
   测试: 2022 (365天 → 349样本)
       ↓
6. 滑窗采样 (NWPacificDataset)
   每个样本: 14天输入 + 3天输出 = 17天窗口
   __getitem__ 实时切片, 不复制数据
       ↓
7. DataLoader → (B, 14, 161, 241, 3), (B, 3, 161, 241, 1), mask
```

**标准化统计量**：首次运行时保存为 `normalization_stats.npz`，后续直接加载。

---

## 3. 模型架构总览

```
输入 (B, 14, 161, 241, 3)           14天 × 161lat × 241lon × 3通道
   │
   ▼
┌──────────────────────────────────────┐
│ InitialEncoder                       │
│  Conv2D×3 → PatchMerging3D(1,4,4)    │  H:161→41, W:241→61, C:3→64
└──────────────────────────────────────┘
   │ (B, 14, 41, 61, 64)
   ▼
┌──────────────────────────────────────┐
│ Encoder PosEmbed (t+h+w)             │  可学习位置嵌入
└──────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│ Encoder Block 0  [axial, dim=64]     │
│  2 × CuboidSelfAttentionLayer        │  轴向注意力
│    + PositionwiseFFN (hidden=256)    │
└──────────────────────────────────────┘
   │ (B, 14, 41, 61, 64)  → mem[0]
   ▼ PatchMerge(1,2,2)  H:41→21, W:61→31
   │ (B, 14, 21, 31, 128)
   ▼
┌──────────────────────────────────────┐
│ Encoder Block 1  [spatial_lg_8, dim=128] │
│  2 × CuboidSelfAttentionLayer        │  空间局部+膨胀注意力
│    + PositionwiseFFN (hidden=512)    │
└──────────────────────────────────────┘
   │ (B, 14, 21, 31, 128)  → mem[1]
   ▼ PatchMerge(1,2,2)  H:21→11, W:31→16
   │ (B, 14, 11, 16, 256)
   ▼
┌──────────────────────────────────────┐
│ Encoder Block 2  [divided_st, dim=256] │
│  2 × CuboidSelfAttentionLayer        │  时空分离注意力
│    + PositionwiseFFN (hidden=1024)   │
└──────────────────────────────────────┘
   │ (B, 14, 11, 16, 256)  → mem[2]
   │
   │  多尺度记忆: mem[0], mem[1], mem[2]
   │  全局向量: 8 × 64维, 逐层传播更新
   │
   ▼
┌──────────────────────────────────────┐
│ Decoder Init (z_init=zeros)          │
│  zeros(3, 11, 16, 256) + PosEmbed    │
│  → z_proj: Linear(256, 256)         │
└──────────────────────────────────────┘
   │ (B, 3, 11, 16, 256)
   ▼
┌──────────────────────────────────────┐
│ Decoder Block 2 (最深层)              │
│  dec_use_first_self_attn=False        │  先 cross → mem[2]
│  cross_attn: cross_1x1               │
│  2 × (self_attn: divided_st + FFN)   │
└──────────────────────────────────────┘
   │ Upsample3DLayer: (11,16)→(21,31), 256→128
   ▼
┌──────────────────────────────────────┐
│ Decoder Block 1                      │
│  2 × (self_attn: spatial_lg_8 +      │
│       cross_attn: cross_1x1 → mem[1])│
└──────────────────────────────────────┘
   │ Upsample3DLayer: (21,31)→(41,61), 128→64
   ▼
┌──────────────────────────────────────┐
│ Decoder Block 0 (最浅层)              │
│  2 × (self_attn: axial +             │
│       cross_attn: cross_1x1 → mem[0])│
└──────────────────────────────────────┘
   │ (B, 3, 41, 61, 64)
   ▼
┌──────────────────────────────────────┐
│ FinalDecoder                         │
│  Upsample3DLayer → Conv2D×2          │  恢复空间: 41→161, 61→241
│  (41,61)→(161,241)                   │
└──────────────────────────────────────┘
   │ (B, 3, 161, 241, 64)
   ▼
┌──────────────────────────────────────┐
│ dec_final_proj: Linear(64, 1)        │  逐像素投影到1通道
└──────────────────────────────────────┘
   │
   ▼
输出 (B, 3, 161, 241, 1)             3天预测 SSTA
```

---

## 4. 初始卷积编码 (InitialEncoder)

**源码**: `cuboid_transformer.py:2442-2520`

**配置**:
```yaml
initial_downsample_type: "conv"
initial_downsample_scale: [1, 4, 4]     # H:4×, W:4×, T:不变
initial_downsample_conv_layers: 3       # 3层Conv2D
initial_downsample_activation: "leaky"
```

**详细流程**:

```
输入: (B, 14, 161, 241, 3)

Step 1: 展平时空维度
    reshape → (B×14, 161, 241, 3)
    permute → (B×14, 3, 161, 241)   # (N, C, H, W)

Step 2: Conv2D×3 (K=3, S=1, P=1, GroupNorm(16)+LeakyReLU)
    Conv0: 3 → 64, GroupNorm, LeakyReLU
    Conv1: 64 → 64, GroupNorm, LeakyReLU
    Conv2: 64 → 64, GroupNorm, LeakyReLU
    空间尺寸保持: 161×241

Step 3: 恢复时空格式
    permute → (B×14, 161, 241, 64)
    reshape → (B, 14, 161, 241, 64)

Step 4: PatchMerging3D(1,4,4)
    H: 161→41 (pad 3 → 164/4=41)
    W: 241→61 (pad 3 → 244/4=61)
    通道: 不变 (out_dim=base_units=64)
    
输出: (B, 14, 41, 61, 64)
```

**为何用 [1,4,4] 而非 [1,1,2]**：161×241 是 24×48 的 34× 倍，需要大幅空间压缩。4×4 下采样将 ~39k 格点压缩至 ~2.5k，使后续 Transformer 的注意力计算量可控。

---

## 5. 位置编码 (PosEmbed)

**配置**: `pos_embed_type: "t+h+w"`

三个独立的可学习 Embedding 表，分别对应时间、高度（纬度）、宽度（经度）三个轴：

```python
T_embed: nn.Embedding(maxT=14, embed_dim=64)
H_embed: nn.Embedding(maxH=41, embed_dim=64)
W_embed: nn.Embedding(maxW=61, embed_dim=64)

前向传播:
    x = x + T_embed([0..13]).reshape(14, 1, 1, 64)
          + H_embed([0..40]).reshape(1, 41, 1, 64)
          + W_embed([0..60]).reshape(1, 1, 61, 64)
```

三个嵌入在 64 维空间相加，每个时空格点获得唯一的编码。编码器中只加一次，解码器中每层上采样后可选重新加（`dec_hierarchical_pos_embed: true`）。

---

## 6. 编码器 — 3层层次结构

**源码**: `CuboidTransformerEncoder`, `cuboid_transformer.py:1649-1923`

**配置**:
```yaml
enc_depth: [2, 2, 2]    # 3层, 每层2个attention block
downsample: 2           # 块间 PatchMerge (1,2,2)
base_units: 64          # 基础通道数, 每层翻倍
```

**通道数自动计算** (`scale_alpha: 1.0`):
```
Block 0: 64
Block 1: 64 × 2¹ = 128
Block 2: 64 × 2² = 256
```

**分辨率变化**:
```
输入编码器: (14, 41, 61, 64)
  Block 0 (axial):         (14, 41, 61, 64)   → mem[0]
  PatchMerge(1,2,2):       (14, 21, 31, 128)
  Block 1 (spatial_lg_8):  (14, 21, 31, 128)  → mem[1]
  PatchMerge(1,2,2):       (14, 11, 16, 256)
  Block 2 (divided_st):    (14, 11, 16, 256)  → mem[2]
```

**mem_l 多尺度记忆**：每个编码器块的输出都保存，供解码器各层跨注意力使用：
```
mem[0]: (B, 14, 41, 61, 64)   ← 高分辨率, 浅语义, 解码器浅层使用
mem[1]: (B, 14, 21, 31, 128)  ← 中分辨率, 解码器中层使用
mem[2]: (B, 14, 11, 16, 256)  ← 低分辨率, 深语义, 解码器深层使用
```

---

## 7. Cuboid Attention — 逐层策略

**当前配置**:
```yaml
self_pattern:       ["axial", "spatial_lg_8", "divided_st"]
cross_self_pattern: ["axial", "spatial_lg_8", "divided_st"]
cross_pattern:      ["cross_1x1", "cross_1x1", "cross_1x1"]
```

### 7.1 Block 0: axial 注意力

**输入空间**: 41×61

**分解策略**:
```
3 个独立长方体:
    cuboid_1: (14,  1,  1)  → 完整时间轴注意力 (12个token)
    cuboid_2: ( 1, 41,  1)  → 完整纬度轴注意力 (41个token)  
    cuboid_3: ( 1,  1, 61)  → 完整经度轴注意力 (61个token)
```

**目的**: 最大空间网格 (41×61)，轴向分解最省计算。T-轴捕获时间演化，H/W-轴各自捕获南北/东西方向的空间依赖。

**计算量**: O(T² + H² + W²) = O(14² + 41² + 61²) ≈ O(5600)，远小于全注意力 O(14²×41²×61²)。

### 7.2 Block 1: spatial_lg_8 注意力

**输入空间**: 21×31

**分解策略**:
```
3 个长方体:
    cuboid_1: (14,  1,  1)  → 完整时间轴 (local)
    cuboid_2: ( 1,  8,  8)  → 局部8×8窗口 (local, 共~12个窗口)
    cuboid_3: ( 1,  8,  8)  → 膨胀8×8窗口 (dilated, 跨越局部窗口)
```

**目的**: 中等空间网格，局部窗口保纹理（涡旋、锋面），膨胀窗口扩大感受野（跨窗口通信）。

### 7.3 Block 2: divided_st 注意力

**输入空间**: 11×16

**分解策略**:
```
2 个长方体:
    cuboid_1: (14,  1,  1)  → 完整时间轴注意力
    cuboid_2: ( 1, 11, 16)  → 全空间注意力 (176个token)
```

**目的**: 最小空间网格 (11×16=176)，全空间注意力计算量可接受。此时每个 token 对应原图约 15×15 的区域，需要全空间交互捕捉大尺度模态。

### 7.4 跨注意力 (cross_1x1)

**所有三层统一使用**:
```
cuboid_hw = (1, 1)     → 逐像素跨注意力
n_temporal = 1          → 解码器3步各自独立跨注意力
strategy = ('l','l','l') → 全局部
```

**Q来自解码器 (query)**: 3个预测步 × 当前空间分辨率
**K/V来自编码器记忆**: 14步历史编码

---

## 8. 全局向量 (Global Vectors)

**配置**:
```yaml
num_global_vectors: 8              # 8个可学习向量
use_dec_self_global: true          # 解码器自注意力中使用
dec_self_update_global: true       # 解码器可更新全局向量
use_dec_cross_global: true         # 解码器跨注意力中使用
use_global_vector_ffn: true        # 全局向量通过FFN
use_global_self_attn: false        # 全局向量之间不做自注意力
global_dim_ratio: 1                # 全局向量维度 = 基础通道数
```

**机制**:
```
初始化: init_global_vectors = Parameter((8, 64))

编码器前向传播:
    for each block:
        x, global_vecs = attn(x, global_vecs)
        # 各cuboid的Q attend到global_vecs的K/V
        # global_vecs attend回各cuboid → 更新
        # global_vecs在不同cuboid之间传递信息
        
    for each downsample:
        global_vecs = Linear(global_vecs)  # 通道对齐
```

**本质**: 8 个全局向量相当于 8 个"信使"，在各个 cuboid 之间传递长距离信息。Cuboid 内部做 local attention，跨 cuboid 通信通过这 8 个向量中转。相比 full attention 显著降低计算量。

**为何在当前任务开启**: 161×241 大网格下，cuboid 数量多（轴向分解产生 ~60 个 cuboid），跨 cuboid 通信需求大，全局向量提供基础的长距离连接能力。

---

## 9. 解码器

**源码**: `CuboidTransformerDecoder`, `cuboid_transformer.py:2087-2440`

**配置**:
```yaml
dec_depth: [2, 2, 2]
dec_use_first_self_attn: false    # 最深层先cross后self
dec_hierarchical_pos_embed: true  # 每层上采样后重加位置编码
```

**解码流程** (从上到下, 3个 block):

```
初始化: zeros(3, 11, 16, 256) + PosEmbed → z_proj→Linear(256,256)

Block 2 (i=2, 最深层, T=3, H=11, W=16, dim=256):
    dec_use_first_self_attn=false:
        Layer 0: cross_attn(cross_1x1) → mem[2] (14,11,16,256)
        然后 self_attn(divided_st)
        Layer 1: self_attn → cross_attn
    输出: (3, 11, 16, 256)

↓ Upsample3DLayer: (11,16)→(21,31), 256→128

Block 1 (i=1, T=3, H=21, W=31, dim=128):
    Layer 0: self_attn(spatial_lg_8) → cross_attn → mem[1] (14,21,31,128)
    Layer 1: 同上
    输出: (3, 21, 31, 128)

↓ Upsample3DLayer: (21,31)→(41,61), 128→64

Block 0 (i=0, 最浅层, T=3, H=41, W=61, dim=64):
    Layer 0: self_attn(axial) → cross_attn → mem[0] (14,41,61,64)
    Layer 1: 同上
    输出: (3, 41, 61, 64)
```

**上采样机制** (Upsample3DLayer):
```
输入: (B, T, H, W, C)
  reshape(B×T, H, W, C) → permute(B×T, C, H, W)
  → nn.Upsample(nearest, size=(H_target, W_target))
  → Conv2D(3×3, C_in→C_out)
  → permute + reshape → (B, T, H_target, W_target, C_out)
```

使用 `size=` 而非 `scale_factor=`，因此可以处理非整数倍率上采样 (11→21, 41→161 等)。

---

## 10. 最终上采样 + 投影

**FinalDecoder 源码**: `cuboid_transformer.py:2522-2580`

**配置**:
```yaml
final_upsample_conv_layers: 2    # 2层Conv2D精细化
```

**流程**:
```
输入: (B, 3, 41, 61, 64)

Step 1: Upsample3DLayer
    目标: (3, 161, 241)
    nearest upsampling + Conv2D(3×3)
    → (B, 3, 161, 241, 64)

Step 2: Conv2D×2 (GroupNorm+LeakyReLU)
    reshape(B×3, 161, 241, 64) → permute → conv_block → permute回
    → (B, 3, 161, 241, 64)

Step 3: 最终投影
    dec_final_proj: Linear(64, 1)
    → (B, 3, 161, 241, 1)
```

**注意**: `CuboidTransformerModel.__init__` 有断言 `H_in == H_out and W_in == W_out`。由于输入输出分辨率相同 (161×241)，此断言不触发。

---

## 11. 损失函数

```python
B, T = pred.shape[0], pred.shape[1]
mask_t = mask.reshape(B, 1, mask.shape[1], mask.shape[2], 1)
loss = ((pred - Y) ** 2 * mask_t).sum() / (mask.sum() * T)
```

**逐步解析**:

1. `(pred - Y)²` — 所有格点的平方误差，包含海陆
2. `* mask_t` — 陆地格点权重清零，只保留海洋
3. `.sum()` — 对全部 (B, T, H, W, C) 求和
4. `/ mask.sum()` — 除以**每个样本的海洋格点数**
5. `/ T` — 除以**预测天数**，得到每像素每天的均方误差

**计算示例** (单样本):
```
海洋格点: 34,878 (161×241 中约 90%)
单样本总误差: 34,878 × 3天 × 1通道 = 104,634 个误差项
loss = sum(误差项) / 34,878 / 3 = 单像素单日平均MSE
```

---

## 12. 评估指标

### 12.1 训练/验证时 (归一化空间)

```python
self.valid_mse(pred_ocean, Y_ocean)  # torchmetrics.MeanSquaredError
self.valid_mae(pred_ocean, Y_ocean)  # torchmetrics.MeanAbsoluteError
```

### 12.2 测试时 (分天指标, 转换°C)

```python
# 累计每天的平方误差和绝对误差
sq_err = ((pred - Y) ** 2 * mask_t).sum(dim=(0,2,3,4))  # (3,) 每天独立
abs_err = ((pred - Y).abs() * mask_t).sum(dim=(0,2,3,4))

# 转换为°C
ssta_std = 0.85  # 从 normalization_stats.npz 读取
rmse_degC = sqrt(mse_norm * ssta_std²)
mae_degC  = mae_norm * ssta_std
```

**输出示例**:
```
  Day   MSE( norm )    MAE( norm )   RMSE(°C)    MAE(°C)
    1     0.252000       0.320000      0.427       0.272
    2     0.309000       0.352000      0.473       0.299
    3     0.361000       0.380000      0.511       0.323
  avg     0.307333       0.350667      0.471       0.298
```

Day1→Day3 误差递增符合预期（远期更难预测）。

---

## 13. 优化器与学习率调度

### 13.1 AdamW

```
lr = 1e-4, weight_decay = 1e-5

参数分组:
    Group 1 (有衰减): 非LayerNorm参数,非bias → weight_decay = 1e-5
    Group 2 (无衰减): LayerNorm参数 + bias → weight_decay = 0
```

### 13.2 两阶段学习率

```
Phase 1 — Warmup (前10%步数):
    lr: 0 → 1e-4 (线性增长)

Phase 2 — Cosine Annealing (后90%步数):
    lr: 1e-4 → 1e-7 (余弦退火)

总步数 = max_epochs × num_train_samples / total_batch_size
       = 30 × 7289 / 16 ≈ 13,667 steps
```

### 13.3 梯度裁剪

```yaml
gradient_clip_val: 1.0    # 最大范数
```

---

## 14. 完整数据形状流转表

| 阶段 | 形状 | 说明 |
|------|------|------|
| 原始 SST | (8035, 321, 561) | ERA5 原始数据 K |
| 原始 Wind | (8035, 321, 561, 2) | u10, v10 m/s |
| 预处理后 SST | (8035, 161, 241) | 裁剪 120-180E, 10-50N, °C |
| 预处理后 Wind | (8035, 161, 241, 2) | 同上 |
| 气候态 | (366, 161, 241) | 逐日22年气候态 |
| SSTA | (8035, 161, 241) | 异常值 °C |
| 合并+归一化后 | (8035, 161, 241, 3) | ssta+u10+v10, z-score |
| 训练数据段 | (7305, 161, 241, 3) | 2001-2020 |
| 验证数据段 | (365, 161, 241, 3) | 2021 |
| 测试数据段 | (365, 161, 241, 3) | 2022 |
| 单样本 | (17, 161, 241, 3) | 14输入+3输出 |
| 模型输入 X | (B, 14, 161, 241, 3) | 批次化 |
| 模型输出 Y_true | (B, 3, 161, 241, 1) | 仅SSTA |
| InitialEncoder后 | (B, 14, 41, 61, 64) | 4×4下采样 |
| Enc Block 0 后 | (B, 14, 41, 61, 64) | mem[0] |
| PatchMerge 后 | (B, 14, 21, 31, 128) | 2×2下采样 |
| Enc Block 1 后 | (B, 14, 21, 31, 128) | mem[1] |
| PatchMerge 后 | (B, 14, 11, 16, 256) | 2×2下采样 |
| Enc Block 2 后 | (B, 14, 11, 16, 256) | mem[2] |
| Decoder 初始化 | (B, 3, 11, 16, 256) | 零向量+位置编码 |
| Dec Block 2 后 | (B, 3, 11, 16, 256) | cross→mem[2] |
| Upsample 后 | (B, 3, 21, 31, 128) | |
| Dec Block 1 后 | (B, 3, 21, 31, 128) | cross→mem[1] |
| Upsample 后 | (B, 3, 41, 61, 64) | |
| Dec Block 0 后 | (B, 3, 41, 61, 64) | cross→mem[0] |
| FinalDecoder 后 | (B, 3, 161, 241, 64) | 恢复原始分辨率 |
| 最终投影 | (B, 3, 161, 241, 1) | |
| 掩码 | (161, 241, 1) | 1=海洋, 0=陆地 |

---

## 15. 配置说明

**文件**: `scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml`

### 关键参数速查

| 分类 | 参数 | 值 | 说明 |
|------|------|-----|------|
| **数据** | `data_dir` | `datasets/SST-PREDICT/` | |
| | `in_len` | 14 | 输入天数 |
| | `out_len` | 3 | 输出天数 |
| **模型** | `base_units` | 64 | 基础通道数 |
| | `enc_depth` | `[2,2,2]` | 3层, 每层2块 |
| | `dec_depth` | `[2,2,2]` | 同上 |
| | `num_global_vectors` | 8 | 全局向量数 |
| | `initial_downsample_scale` | `[1,4,4]` | 空间16×压缩 |
| | `num_heads` | 4 | 注意力头数 |
| **注意力** | `self_pattern` | `["axial","spatial_lg_8","divided_st"]` | 逐层不同 |
| | `cross_pattern` | `["cross_1x1"]*3` | 像素级跨注意力 |
| **优化** | `lr` | 1e-4 | 学习率 |
| | `total_batch_size` | 16 | 有效batch |
| | `micro_batch_size` | 2 | 每卡batch |
| | `max_epochs` | 30 | |
| | `early_stop` | true | patience=10 |
| **评估** | `save_top_k` | 3 | 保留最优3个模型 |

---

## 16. 训练流程

### 16.1 实验目录结构

```
experiments/nwp_exp1/
├── hparams.json          ← 超参数 (JSON, 训练开始时生成)
├── metrics.csv           ← 每轮指标 (CSV, 逐轮追加)
│   列: epoch, train_loss, valid_loss, valid_mse, valid_mae, lr
├── test_metrics.csv      ← 测试分天指标 (°C)
├── cfg.yaml              ← 配置文件备份
└── checkpoints/
    ├── model-epoch=xxx.ckpt   ← 最优N个 (优化器+模型+LR)
    ├── last.ckpt              ← 最新 (断点续训用)
    └── best_model.pt          ← 纯模型权重 (推理用)
```

**不再有** `lightning_logs/version_N/` 目录 — 所有指标直接写入 CSV。

### 16.2 运行命令

```bash
# 训练
python scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py \
    --gpus 1 --save nwp_exp1 --data_dir datasets/SST-PREDICT/ \
    --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml

# 断点续训
python scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py \
    --gpus 1 --save nwp_exp1 --data_dir datasets/SST-PREDICT/ \
    --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml \
    --ckpt_name last.ckpt

# 测试
python scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py \
    --gpus 1 --test --save nwp_exp1 --data_dir datasets/SST-PREDICT/ \
    --ckpt_name last.ckpt
```

---

## 附录: 关键文件速查

| 文件 | 内容 |
|------|------|
| `src/earthformer/cuboid_transformer/cuboid_transformer.py` | 完整模型: CuboidAttention, Encoder, Decoder, CuboidTransformerModel |
| `src/earthformer/cuboid_transformer/cuboid_transformer_patterns.py` | 注意力模式注册表 (axial, spatial_lg, divided_st 等) |
| `src/earthformer/cuboid_transformer/utils.py` | RMSNorm, padding, 位置嵌入, 初始化 |
| `src/earthformer/datasets/nw_pacific_dataset.py` | 数据加载与 Dataset 构建 |
| `scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py` | 训练入口 |
| `scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml` | 训练配置 |
| `scripts/cuboid_transformer/nwp_sst/visualize_logs.py` | 训练曲线可视化 |
| `scripts/datasets/preprocess_nwp.py` | 空间裁剪+单位转换 |
| `scripts/datasets/generate_ocean_mask.py` | 海陆掩码生成 |
| `scripts/datasets/compute_climatology.py` | 逐日气候态计算 |
| `scripts/datasets/compute_ssta.py` | SSTA 计算 |
