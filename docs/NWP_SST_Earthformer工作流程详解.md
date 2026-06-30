# Earthformer 西北太平洋 SSTA 预测 — 工作流程详解

> 基于 `scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py` 和 `cfg_nwp.yaml` 配置
> 最后更新: 预测7天, 4通道输入, scale_alpha=0.5, 含频率分支 (SpatialFrequencyBranch)

---

## 目录

1. [任务定义](#1-任务定义)
2. [数据管线](#2-数据管线)
3. [模型架构总览](#3-模型架构总览)
4. [初始卷积编码 (InitialEncoder)](#4-初始卷积编码)
5. [空间频率分支 (SpatialFrequencyBranch)](#5-空间频率分支)
6. [位置编码 (PosEmbed)](#6-位置编码)
7. [编码器 — 3层层次结构](#7-编码器)
8. [Cuboid Attention 逐层策略](#8-cuboid-attention)
9. [全局向量 (Global Vectors)](#9-全局向量)
10. [解码器](#10-解码器)
11. [最终上采样 + 投影](#11-最终上采样)
12. [损失函数](#12-损失函数)
13. [评估指标](#13-评估指标)
14. [优化器与学习率调度](#14-优化器与学习率调度)
15. [完整数据形状流转表](#15-数据形状流转表)
16. [配置说明](#16-配置说明)
17. [训练流程](#17-训练流程)
18. [Persistence Baseline](#18-persistence-baseline)

---

## 1. 任务定义

| 项目 | 值 |
|------|-----|
| 输入 | 14 天 × 161×241 × 4 通道 [ssta, u10, v10, sla] |
| 输出 | 7 天 × 161×241 × 1 通道 [ssta] |
| 区域 | 10°N–50°N, 120°E–180°E |
| 分辨率 | 0.25° |
| 数据源 | ERA5 (SST/Wind) + CMEMS AVISO (SLA), 2001-2025 日数据 |
| 模型参数量 | ~10M (scale_alpha=0.5, [1,2,2] downsampling) |

**与原始 ENSO 任务的关键区别**：

| | ENSO (原) | NWP SSTA (当前) |
|------|----------|-----|
| 网格 | 24×48 | 161×241 (34×更大) |
| 输入长度 | 12 月 | 14 天 |
| 输出长度 | 26 月 | 7 天 |
| 输入通道 | 4 (sst, t300, ua, va) | 4 (ssta, u10, v10, sla) |
| 输出通道 | 4 | 1 (仅 ssta) |
| 编码器层数 | 2 层 | 3 层 |
| 注意力策略 | 全 axial | 逐层不同 |
| 全局向量 | 关闭 | 开启 (8 个) |
| 初始下采样 | [1,1,2] | [1,2,2] (可变) |
| scale_alpha | — | 0.5 |
| 参数量 | ~7M | ~10M |
| 步长 | 1 | 3 (训练集) |
| 评估 | Niño 相关系数 | 逐像素 SSTA MSE/MAE/RMSE (°C) |

---

## 2. 数据管线

### 2.1 数据来源

ERA5 (SST/Wind) + CMEMS AVISO (SLA)，经过以下预处理步骤（各脚本独立运行）：

| 步骤 | 脚本 | 输入 → 输出 |
|------|------|-----------|
| 空间裁剪+单位转换 | `preprocess_nwp.py` | SST.nc + wind01-22.nc → SST_cropped.nc + Wind_cropped.nc |
| 海陆掩码 | `generate_ocean_mask.py` | SST_cropped.nc → mask.npy |
| 逐日气候态 | `compute_climatology.py` | SST_cropped.nc → climatology.nc |
| SSTA 计算 | `compute_ssta.py` | SST_cropped.nc + climatology.nc → ssta.nc |
| SLA 预处理 | `preprocess_sla.py` | sladata/ 四子目录 → SLA_cropped.nc |

### 2.2 训练时数据加载

**文件**: `src/earthformer/datasets/nw_pacific_dataset.py`

**函数**: `build_dataloaders(data_dir, batch_size, num_workers)`

**处理流程**：

```
1. 读取 ssta.nc + Wind_cropped.nc + SLA_cropped.nc + mask.npy
       ↓
2. 归一化 (仅用训练集 2001-2022 计算 mean/std)
   ssta = (ssta - mean) / std
   u10  = (u10  - mean) / std
   v10  = (v10  - mean) / std
   sla  = (sla  - mean) / std       ← 新增, NaN-safe
       ↓
3. 掩码陆地 (归一化后) → 陆地值 ≡ 0
   ssta = np.where(mask, ssta, 0.0)
   u10  = np.where(mask, u10,  0.0)
   v10  = np.where(mask, v10,  0.0)
   sla  = np.where((mask>0) & ~np.isnan(sla), sla, 0.0)  ← 双重掩码
       ↓
4. 堆叠通道
   data = stack([ssta, u10, v10, sla], axis=-1)  → (9131, 161, 241, 4)
       ↓
5. 按年切分
   训练: 2001-2022 (8035天 → ~2672样本, stride=3)
   验证: 2023-2024 (731天  → ~715样本, stride=1)
   测试: 2025        (365天  → ~349样本, stride=1)
       ↓
6. 滑窗采样 (NWPacificDataset, stride=3 for train)
   每个样本: 14天输入 + 7天输出 = 21天窗口
   stride=3: 训练样本每隔3天采样一次, 减少重叠
   __getitem__ 实时切片, 不复制数据
       ↓
7. DataLoader → (B, 14, 161, 241, 4), (B, 7, 161, 241, 1), mask
```

**标准化统计量**：首次运行时保存为 `normalization_stats.npz`，后续直接加载。含 4 通道统计: `ssta_mean/std`, `u10_mean/std`, `v10_mean/std`, `sla_mean/std`。

---

## 3. 模型架构总览

> 以下形状基于 `scale_alpha: 0.5` + `initial_downsample_scale: [1,2,2]`。
> 不同配置下具体 dim 和 spatial size 会变化，原则不变。

```
输入 (B, 14, 161, 241, 4)           14天 × 161lat × 241lon × 4通道
   │
   ▼
┌──────────────────────────────────────┐
│ InitialEncoder                       │
│  Conv2D×3 → PatchMerging3D(1,2,2)    │  H:161→81, W:241→121, C:4→64
└──────────────────────────────────────┘
   │ (B, 14, 81, 121, 64)
   ▼
┌──────────────────────────────────────┐
│ SpatialFrequencyBranch               │  FFT → amp gating → IFFT
│  State-conditioned频段Gating          │  6K params, alpha init=0
└──────────────────────────────────────┘
   │ (B, 14, 81, 121, 64)
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
   │ (B, 14, 81, 121, 64)  → mem[0]
   ▼ PatchMerge(1,2,2)  H:81→41, W:121→61
   │ (B, 14, 41, 61, ~90)
   ▼
┌──────────────────────────────────────┐
│ Encoder Block 1  [spatial_lg_8, dim≈90] │
│  2 × CuboidSelfAttentionLayer        │  空间局部+膨胀注意力
│    + PositionwiseFFN (hidden≈360)    │
└──────────────────────────────────────┘
   │ (B, 14, 41, 61, ~90)  → mem[1]
   ▼ PatchMerge(1,2,2)  H:41→21, W:61→31
   │ (B, 14, 21, 31, 128)
   ▼
┌──────────────────────────────────────┐
│ Encoder Block 2  [divided_st, dim=128] │
│  2 × CuboidSelfAttentionLayer        │  时空分离注意力
│    + PositionwiseFFN (hidden=512)    │
└──────────────────────────────────────┘
   │ (B, 14, 21, 31, 128)  → mem[2]
   │
   │  多尺度记忆: mem[0], mem[1], mem[2]
   │  全局向量: 8 × 64维, 逐层传播更新
   │
   ▼
┌──────────────────────────────────────┐
│ Decoder Init (z_init=zeros)          │
│  zeros(7, 21, 31, 128) + PosEmbed    │
│  → z_proj: Linear(128, 128)         │
└──────────────────────────────────────┘
   │ (B, 7, 21, 31, 128)
   ▼
┌──────────────────────────────────────┐
│ Decoder Block 2 (最深层)              │
│  dec_use_first_self_attn=False        │  先 cross → mem[2]
│  cross_attn: cross_1x1               │  T_dec=7 vs T_enc=14
│  2 × (self_attn: divided_st + FFN)   │
└──────────────────────────────────────┘
   │ Upsample3DLayer: (21,31)→(41,61), 128→~90
   ▼
┌──────────────────────────────────────┐
│ Decoder Block 1                      │
│  2 × (self_attn: spatial_lg_8 +      │
│       cross_attn: cross_1x1 → mem[1])│
└──────────────────────────────────────┘
   │ Upsample3DLayer: (41,61)→(81,121), ~90→64
   ▼
┌──────────────────────────────────────┐
│ Decoder Block 0 (最浅层)              │
│  2 × (self_attn: axial +             │
│       cross_attn: cross_1x1 → mem[0])│
└──────────────────────────────────────┘
   │ (B, 7, 81, 121, 64)
   ▼
┌──────────────────────────────────────┐
│ FinalDecoder                         │
│  Upsample3DLayer → Conv2D×2          │  恢复空间: 81→161, 121→241
└──────────────────────────────────────┘
   │ (B, 7, 161, 241, 64)
   ▼
┌──────────────────────────────────────┐
│ dec_final_proj: Linear(64, 1)        │  逐像素投影到SSTA
└──────────────────────────────────────┘
   │
   ▼
输出 (B, 7, 161, 241, 1)             7天预测 SSTA
```

---

## 4. 初始卷积编码 (InitialEncoder)

**源码**: `cuboid_transformer.py`

**配置**:
```yaml
initial_downsample_type: "conv"
initial_downsample_scale: [1, 2, 2]     # H:2×, W:2×, T:不变
initial_downsample_conv_layers: 3       # 3层Conv2D
initial_downsample_activation: "leaky"
```

**详细流程**:

```
输入: (B, 14, 161, 241, 4)

Step 1: 展平时空维度
    reshape → (B×14, 161, 241, 4)
    permute → (B×14, 4, 161, 241)   # (N, C, H, W)

Step 2: Conv2D×3 (K=3, S=1, P=1, GroupNorm(16)+LeakyReLU)
    Conv0: 4 → 64, GroupNorm, LeakyReLU      ← 第4通道(SLA) kernel 随机初始化
    Conv1: 64 → 64, GroupNorm, LeakyReLU
    Conv2: 64 → 64, GroupNorm, LeakyReLU
    空间尺寸保持: 161×241

Step 3: 恢复时空格式
    permute → (B×14, 161, 241, 64)
    reshape → (B, 14, 161, 241, 64)

Step 4: PatchMerging3D(1,2,2)
    H: 161→81 (pad 1 → 162/2=81)
    W: 241→121 (pad 1 → 242/2=121)
    通道: 不变 (out_dim=base_units=64)

输出: (B, 14, 81, 121, 64)
```

---

## 5. 空间频率分支 (SpatialFrequencyBranch)

**源码**: `src/earthformer/cuboid_transformer/spatial_frequency_branch.py`

**插入位置**: `InitialEncoder` 之后、`enc_pos_embed` 之前

```python
# cuboid_transformer.py forward()
x = self.initial_encoder(x)       # (B, T, H, W, D)
x = self.freq_branch(x)           # <-- 频率增强
x = self.enc_pos_embed(x)
```

### 5.1 设计动机

西北太平洋 SSTA 预测涉及多种具有清晰频域物理签名的海洋现象:
- **低频**: ENSO 遥相关、季节循环、黑潮大弯曲
- **中频**: 中尺度涡旋、锋面、Rossby 波
- **高频**: 小尺度混合、观测噪声

Cuboid Transformer 通过注意力隐式学习这些模式，但缺乏显式的频域结构表示。本模块提供 **~6K 参数（<0.06%）** 的轻量频域增强。

### 5.2 架构

```
x: (B, T, H, W, D)
  │
  ├─────────────────────────────────┐
  │                                 │
  ▼                                 │
rFFT2(H, W)                         │  2D空间实FFT
  │                                 │
  ▼                                 │
X_f: (B, T, H, W_f, D) complex      │
  ├── amp  = |X_f|                  │  振幅（参与学习）
  └── phase = ∠X_f                  │  相位（保留不变）
  │                                 │
  ▼                                 │
┌─ State Conditioner ────────────┐  │
│ x → chunk(4组, dim=-1)         │  │  4组×16通道分别池化
│   → per-group GlobalAvgPool    │  │  捕获流域级物理状态
│   → Linear(16→4) × 4           │  │
│   → concat → Linear(16→K)      │  │
│   → softmax → band_w: (B, K)   │  │  每样本动态频段分配
└────────────────────────────────┘  │
  │                                 │
  ▼                                 │
┌─ Spectral Gating ──────────────┐  │
│ profile[k,h,w]: (K,H,W_f)      │  │  可学习频段空间签名
│ gate[k,d]:      (K,D)          │  │  每频段每通道门控
│                                 │  │
│ logits[b,k,h,w,d] =            │  │
│   band_w[b,k]                  │  │
│   × profile[k,h,w]             │  │
│   × σ(gate[k,d])               │  │
│                                 │  │
│ attn = softmax_k(logits)        │  │  频段间归一化
│ weight[b,h,w,d] =              │  │
│   Σ_k attn × profile[k,h,w]    │  │  有界增强权重
└────────────────────────────────┘  │
  │                                 │
  ▼                                 │
amp' = amp × weight                 │  频域直接gating
  │                                 │
  ▼                                 │
X' = amp' × exp(i × phase)          │  原相位重建
  │                                 │
  ▼                                 │
irfft2(X', s=(H,W))                 │  回到空域
  │                                 │
  ▼                                 │
  × α (learnable, init=0)           │  可学习缩放
  │                                 │
  └─────────────────────────────────┘
  │
  ▼
output = x + α × freq_out           (B, T, H, W, D)
```

### 5.3 核心设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| FFT维度 | 仅2D空间 (H,W) | 14天时间窗口太短, 时间FFT频率分辨率极低 |
| 处理对象 | 仅振幅, 不碰相位 | 相位训练初期极易不稳定 |
| 频率分配 | State-conditioned | ENSO/涡旋/正常态需要不同的频段增强 |
| 通道分组 | 4组独立池化 | SSTA/U10/V10/SLA 频谱结构不同, 不应混合 |
| 归一化 | softmax频段间 | 有界稳定, 天然频率守恒 |
| 初始化 | α=0, gates=-2.0 | 训练初期等价于无分支, 安全插入已有checkpoint |
| 正则化 | Entropy loss (1e-4) | 防single-band collapse |

### 5.4 代码结构

```python
class SpatialFrequencyBranch(nn.Module):
    def __init__(self, dim=64, num_bands=4, num_groups=4, state_hidden=16):
        self.group_pools   # 4×Linear(16→4) per-group池化
        self.state_mlp     # Linear(16→4) band权重
        self.freq_profiles # (4, 1, 1) → bilinear插值到(H,W_f)
        self.channel_gates # (4, 64) 每频段每通道门控
        self.alpha         # scalar, init=0

    def forward(self, x):
        # rFFT2 → |amp|, phase
        # State conditioner → band_w (B, 4)
        # Spectral gating → softmax → weight
        # amp × weight → irfft2 → alpha * residual
        return x + self.alpha * freq_out

    def entropy_loss(self, x):
        # 正则化: log(K) - H(band_w)
```

### 5.5 训练时的 Entropy 正则化

在 `training_step` 中:

```python
loss = mse_loss
entropy_reg = self.torch_nn_module.freq_branch.entropy_loss(
    self.torch_nn_module._freq_input)
loss = loss + 1e-4 * entropy_reg
```

记录在 `metrics.csv` 中可监控频段分布是否 collapse。

### 5.6 论文可解释性

1. **可视化 freq_profiles**: 训练后 4 个 profile 自然收敛到低→高频的渐近分布, 边界由数据决定
2. **band_w 分析**: ENSO 位相样本 vs 正常年样本的 band 激活模式对比
3. **Channel gate**: 打印 σ(gate[k,:]) 查看各频段偏好哪些通道

---

## 6. 位置编码 (PosEmbed)

**配置**: `pos_embed_type: "t+h+w"`

三个独立的可学习 Embedding 表，分别对应时间、高度（纬度）、宽度（经度）三个轴：

```python
T_embed: nn.Embedding(maxT=14, embed_dim=dim)
H_embed: nn.Embedding(maxH=H,  embed_dim=dim)
W_embed: nn.Embedding(maxW=W,  embed_dim=dim)

前向传播:
    x = x + T_embed([0..13]).reshape(14, 1, 1, dim)
          + H_embed([0..H-1]).reshape(1, H, 1, dim)
          + W_embed([0..W-1]).reshape(1, 1, W, dim)
```

encoder 中只加一次，decoder 中每层上采样后可选重新加（`dec_hierarchical_pos_embed: true`）。

---

## 7. 编码器 — 3层层次结构

**源码**: `CuboidTransformerEncoder`, `cuboid_transformer.py`

**配置**:
```yaml
enc_depth: [2, 2, 2]    # 3层, 每层2个attention block
downsample: 2           # 块间 PatchMerge (1,2,2)
base_units: 64          # 基础通道数
scale_alpha: 0.5        # 通道增长速度减半
```

**通道数自动计算** (`scale_alpha: 0.5`):
```
Block 0: int(64 × 2^(0×0.5)) = 64
Block 1: int(64 × 2^(1×0.5)) ≈ 90
Block 2: int(64 × 2^(2×0.5)) = 128
```
> `scale_alpha=1.0` 时通道数为 64→128→256，参数约 18.4M。
> `scale_alpha=0.5` 时降为 64→90→128，参数约 10M，有效缓解过拟合。

**分辨率变化** (以 [1,2,2] initial downsampling 为例):
```
输入编码器: (14, 81, 121, 64)
  Block 0 (axial):         (14, 81, 121, 64)   → mem[0]
  PatchMerge(1,2,2):       (14, 41, 61, ~90)
  Block 1 (spatial_lg_8):  (14, 41, 61, ~90)   → mem[1]
  PatchMerge(1,2,2):       (14, 21, 31, 128)
  Block 2 (divided_st):    (14, 21, 31, 128)   → mem[2]
```

**mem_l 多尺度记忆**：每个编码器块的输出都保存，供解码器各层跨注意力使用：
```
mem[0]: (B, 14, 81, 121, 64)   ← 高分辨率, 浅语义, 解码器浅层使用
mem[1]: (B, 14, 41, 61, ~90)   ← 中分辨率, 解码器中层使用
mem[2]: (B, 14, 21, 31, 128)   ← 低分辨率, 深语义, 解码器深层使用
```

---

## 8. Cuboid Attention — 逐层策略

**当前配置**:
```yaml
self_pattern:       ["axial", "spatial_lg_8", "divided_st"]
cross_self_pattern: ["axial", "spatial_lg_8", "divided_st"]
cross_pattern:      ["cross_1x1", "cross_1x1", "cross_1x1"]
```

### 7.1 Block 0: axial 注意力

**输入空间**: 81×121 (或 41×61, 取决于 initial_downsample_scale)

3 个独立长方体:
- cuboid_1: (14, 1, 1) → 完整时间轴注意力
- cuboid_2: (1, H, 1) → 完整纬度轴注意力
- cuboid_3: (1, 1, W) → 完整经度轴注意力

**目的**: 最大空间网格，轴向分解最省计算。T-轴捕获时间演化，H/W-轴各自捕获空间方向依赖。

### 7.2 Block 1: spatial_lg_8 注意力

3 个长方体:
- cuboid_1: (14, 1, 1) → 完整时间轴 (local)
- cuboid_2: (1, 8, 8) → 局部8×8窗口 (local)
- cuboid_3: (1, 8, 8) → 膨胀8×8窗口 (dilated, 跨窗口通信)

**目的**: 局部窗口保纹理（涡旋、锋面），膨胀窗口扩大感受野。

### 7.3 Block 2: divided_st 注意力

2 个长方体:
- cuboid_1: (14, 1, 1) → 完整时间轴注意力
- cuboid_2: (1, H, W) → 全空间注意力

**目的**: 最小空间网格，全空间注意力 token 数可控。捕捉大尺度气候模态（ENSO、PDO 等）。

### 7.4 跨注意力 (cross_1x1)

**所有三层统一使用**:
```
cuboid_hw = (1, 1)     → 逐像素跨注意力
n_temporal = 1          → 解码器各步独立跨注意力
strategy = ('l','l','l') → 全局部
```

**Q 来自解码器**: T_out 个预测步 × 当前空间分辨率
**K/V 来自编码器记忆**: 14 步历史编码

从 3 天改为 7 天: 解码器 cross-attention 步数从 3→7, 计算量同比增加。

---

## 9. 全局向量 (Global Vectors)

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

**本质**: 8 个全局向量相当于"信使"，在各个 cuboid 之间传递长距离信息。Cuboid 内部做 local attention，跨 cuboid 通信通过这 8 个向量中转。

---

## 10. 解码器

**源码**: `CuboidTransformerDecoder`, `cuboid_transformer.py`

**配置**:
```yaml
dec_depth: [2, 2, 2]
dec_use_first_self_attn: false    # 最深层先cross后self
dec_hierarchical_pos_embed: true  # 每层上采样后重加位置编码
```

**解码流程** (从上到下, 3 个 block, T_out=7):

```
初始化: zeros(7, H3, W3, dim2) + PosEmbed → z_proj→Linear(dim2, dim2)

Block 2 (i=2, 最深层, T=7, dim=128):
    dec_use_first_self_attn=false:
        Layer 0: cross_attn(cross_1x1) → mem[2] (14, H3, W3, 128)
        然后 self_attn(divided_st)
        Layer 1: self_attn → cross_attn
    输出: (7, H3, W3, 128)

↓ Upsample3DLayer: (H3,W3)→(H1,W1), 128→dim1

Block 1 (i=1, T=7, dim=dim1≈90):
    Layer 0: self_attn(spatial_lg_8) → cross_attn → mem[1]
    Layer 1: 同上
    输出: (7, H1, W1, dim1)

↓ Upsample3DLayer: (H1,W1)→(H0,W0), dim1→64

Block 0 (i=0, 最浅层, T=7, dim=64):
    Layer 0: self_attn(axial) → cross_attn → mem[0]
    Layer 1: 同上
    输出: (7, H0, W0, 64)
```

**上采样机制** (Upsample3DLayer):
```
输入: (B, T, H, W, C)
  reshape(B×T, H, W, C) → permute(B×T, C, H, W)
  → nn.Upsample(nearest, size=(H_target, W_target))
  → Conv2D(3×3, C_in→C_out)
  → permute + reshape → (B, T, H_target, W_target, C_out)
```

---

## 11. 最终上采样 + 投影

**配置**:
```yaml
final_upsample_conv_layers: 2    # 2层Conv2D精细化
```

**流程**:
```
输入: (B, 7, H0, W0, 64)

Step 1: Upsample3DLayer
    目标: (7, 161, 241)
    nearest upsampling + Conv2D(3×3)
    → (B, 7, 161, 241, 64)

Step 2: Conv2D×2 (GroupNorm+LeakyReLU)
    reshape(B×7, 161, 241, 64) → permute → conv_block → permute回
    → (B, 7, 161, 241, 64)

Step 3: 最终投影
    dec_final_proj: Linear(64, 1)
    → (B, 7, 161, 241, 1)
```

---

## 12. 损失函数

```python
def training_step(self, batch, batch_idx):
    X, Y, mask = batch
    pred = self(X, mask)
    B, T = pred.shape[0], pred.shape[1]
    mask_t = mask.reshape(B, 1, mask.shape[1], mask.shape[2], 1)

    loss = ((pred - Y) ** 2 * mask_t).sum() / (mask.sum() * B * T)
    #                                          ^^^^^^^^^^^^^^^^^^^
    #                                          /ocean_pixels /B /T

    # Entropy正则化 — 防频率分支band collapse
    entropy_reg = self.torch_nn_module.freq_branch.entropy_loss(
        self.torch_nn_module._freq_input)
    loss = loss + 1e-4 * entropy_reg
    self.log('entropy_reg', entropy_reg, on_step=False, on_epoch=True)
```

**逐步解析**:

1. `(pred - Y)²` — 所有格点 (海洋+陆地) 的平方误差
2. `* mask_t` — 陆地格点权重清零
3. `.sum()` — 对全部 (B, T, H, W, C) 求和
4. `/ mask.sum()` — 除以单样本海洋格点数 (~34,879)
5. `/ B` — 除以 batch size，得到**每样本**
6. `/ T` — 除以预测天数 (7)，得到**每像素每天**

**重要**: 分母含 `B`（于 2026-06 修复），确保 loss 值是 per-sample-per-day-per-pixel 的 MSE，train/val 值的量级不受 batch_size 影响。

**Entropy 正则化**: `loss_entropy = log(K) - H(band_w)`, 权重 1e-4。当 band_w 接近均匀分布时值为 0，collapse 到单一频段时趋近 log(K)。记录在 CSV 的 `entropy_reg` 列中监控。这是全局批次级统计量（`on_step=False`），数值稳定。

---

## 13. 评估指标

### 13.1 训练/验证时 (归一化空间)

```python
self.valid_mse(pred_ocean, Y_ocean)  # torchmetrics.MeanSquaredError
self.valid_mae(pred_ocean, Y_ocean)  # torchmetrics.MeanAbsoluteError
```

torchmetrics 对所有像素（含陆地=0）求均值。`valid_mse_epoch` 用于 checkpoint 选择。

### 13.2 测试时 (分天指标, 转换°C)

```python
# 累计每天的平方误差和绝对误差
sq_err = ((pred - Y) ** 2 * mask_t).sum(dim=(0,2,3,4))   # (T_out,) 每天独立
abs_err = ((pred - Y).abs() * mask_t).sum(dim=(0,2,3,4))  # (T_out,)

# 全局归一化: sum over batches → divide by total ocean pixels
n_total = sum(mask_t.sum() for each batch)  # = total_samples × ocean_pixels_per_sample
mse_per_day = total_sq_err / n_total         # (T_out,) per-pixel MSE

# 转换为°C
mse_degC = mse_per_day × ssta_std²
rmse_degC = sqrt(mse_degC)
mae_degC = mae_per_day × ssta_std
```

**输出示例**（7天预报）:
```
  Day   MSE( norm )    MAE( norm )   RMSE(°C)    MAE(°C)
    1     0.xxxxxx       0.xxxxxx      0.xxxx       0.xxxx
    2     0.xxxxxx       0.xxxxxx      0.xxxx       0.xxxx
    ...
    7     0.xxxxxx       0.xxxxxx      0.xxxx       0.xxxx
  avg     0.xxxxxx       0.xxxxxx      0.xxxx       0.xxxx
```

---

## 14. 优化器与学习率调度

### 13.1 AdamW

```
lr = 1e-4, weight_decay = 1e-4

参数分组:
    Group 1 (有衰减): 非LayerNorm参数,非bias → weight_decay = 1e-4
    Group 2 (无衰减): LayerNorm参数 + bias → weight_decay = 0
```

### 13.2 两阶段学习率

```
Phase 1 — Warmup (前10%步数):
    lr: 0 → 1e-4 (线性增长)

Phase 2 — Cosine Annealing (后90%步数):
    lr: 1e-4 → 1e-7 (余弦退火)

总步数 ≈ max_epochs × num_train_samples / total_batch_size
       ≈ 50 × ~2672 / 16 ≈ 8,350 steps  (stride=3)
```

### 13.3 正则化

```yaml
attn_drop: 0.2       # 注意力 dropout
proj_drop: 0.2       # 投影 dropout
ffn_drop: 0.3        # FFN dropout (最深层FFN参数量大, 用更大dropout)
wd: 1.0e-04          # weight decay
gradient_clip_val: 1.0
early_stop_patience: 10
```

---

## 15. 完整数据形状流转表

> 以下基于 `scale_alpha: 0.5`, `initial_downsample_scale: [1,2,2]`。
> 括号中的值是 `scale_alpha=1.0` / `[1,4,4]` 时的对照。

| 阶段 | 形状 | 说明 |
|------|------|------|
| 原始 SST | (9131, 321, 561) | ERA5 原始数据 K |
| 原始 Wind | (9131, 321, 561, 2) | u10, v10 m/s |
| 原始 SLA | — 720×1440 逐日文件 | CMEMS AVISO 0.25° global, m |
| 预处理后 SST | (9131, 161, 241) | 裁剪 120-180E, 10-50N, °C |
| 预处理后 Wind | (9131, 161, 241, 2) | 同上 |
| 预处理后 SLA | (9131, 161, 241) | 裁剪+双线性插值, m |
| 气候态 | (366, 161, 241) | 逐日22年气候态 |
| SSTA | (9131, 161, 241) | 异常值 °C |
| 合并+归一化后 | (9131, 161, 241, 4) | ssta+u10+v10+sla, z-score, 陆地→0 |
| 训练数据段 | (8035, 161, 241, 4) | 2001-2022 |
| 验证数据段 | (731, 161, 241, 4) | 2023-2024 |
| 测试数据段 | (365, 161, 241, 4) | 2025 |
| 单样本 | (21, 161, 241, 4) | 14输入+7输出 |
| 模型输入 X | (B, 14, 161, 241, 4) | 批次化 |
| 模型输出 Y_true | (B, 7, 161, 241, 1) | 仅SSTA |
| InitialEncoder后 | (B, 14, 81, 121, 64) | 2×2下采样 |
| FreqBranch后 | (B, 14, 81, 121, 64) | 频率增强, 形状不变 |
| Enc Block 0 后 | (B, 14, 81, 121, 64) | mem[0] |
| PatchMerge 后 | (B, 14, 41, 61, ~90) | 2×2下采样 |
| Enc Block 1 后 | (B, 14, 41, 61, ~90) | mem[1] |
| PatchMerge 后 | (B, 14, 21, 31, 128) | 2×2下采样 |
| Enc Block 2 后 | (B, 14, 21, 31, 128) | mem[2] |
| Decoder 初始化 | (B, 7, 21, 31, 128) | 零向量+位置编码 |
| Dec Block 2 后 | (B, 7, 21, 31, 128) | cross→mem[2] |
| Upsample 后 | (B, 7, 41, 61, ~90) | |
| Dec Block 1 后 | (B, 7, 41, 61, ~90) | cross→mem[1] |
| Upsample 后 | (B, 7, 81, 121, 64) | |
| Dec Block 0 后 | (B, 7, 81, 121, 64) | cross→mem[0] |
| FinalDecoder 后 | (B, 7, 161, 241, 64) | 恢复原始分辨率 |
| 最终投影 | (B, 7, 161, 241, 1) | |
| 掩码 | (161, 241, 1) | 1=海洋, 0=陆地 |

---

## 16. 配置说明

**文件**: `scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml`

### 关键参数速查

| 分类 | 参数 | 值 | 说明 |
|------|------|-----|------|
| **数据** | `data_dir` | `datasets/SST-PREDICT/` | |
| | `in_len` | 14 | 输入天数 |
| | `out_len` | 7 | 输出天数 |
| | `var_names` | `[ssta, u10, v10, sla]` | 4通道 |
| **模型** | `base_units` | 64 | 基础通道数 |
| | `scale_alpha` | 0.5 | 通道增长因子 |
| | `enc_depth` | `[2,2,2]` | 3层, 每层2块 |
| | `dec_depth` | `[2,2,2]` | 同上 |
| | `num_global_vectors` | 8 | 全局向量数 |
| | `initial_downsample_scale` | `[1,2,2]` | 空间4×压缩 |
| | `num_heads` | 4 | 注意力头数 |
| | `input_shape` | `[14, 161, 241, 4]` | 4通道输入 |
| | `target_shape` | `[7, 161, 241, 1]` | 7天SSTA输出 |
| **注意力** | `self_pattern` | `["axial","spatial_lg_8","divided_st"]` | 逐层不同 |
| | `cross_pattern` | `["cross_1x1"]*3` | 像素级跨注意力 |
| **正则化** | `attn_drop` | 0.2 | 注意力 dropout |
| | `proj_drop` | 0.2 | 投影 dropout |
| | `ffn_drop` | 0.3 | FFN dropout |
| | `wd` | 1e-4 | weight decay |
| **优化** | `lr` | 1e-4 | 学习率 |
| | `total_batch_size` | 16 | 有效batch (micro_bs×accum) |
| | `micro_batch_size` | 2 | 每卡batch |
| | `max_epochs` | 50 | |
| | `early_stop` | true | patience=10 |
| **数据** | `stride` | 3 | 训练滑窗步长 |
| **评估** | `save_top_k` | 3 | 保留最优3个模型 |

---

## 17. 训练流程

### 16.1 实验目录结构

```
experiments/nwp_7day/
├── hparams.json          ← 超参数 (JSON, 训练开始时生成)
├── metrics.csv           ← 每轮指标 (CSV, 新训练自动清空)
│   列: epoch, train_loss, valid_loss, valid_mse, valid_mae, lr
├── test_metrics.csv      ← 测试分天指标 (°C, 7天)
├── cfg.yaml              ← 配置文件备份
└── checkpoints/
    ├── model-epoch=xxx.ckpt   ← 最优N个 (优化器+模型+LR)
    ├── last.ckpt              ← 最新 (断点续训用)
    └── best_model.pt          ← 纯模型权重 (推理用)
```

### 16.2 运行命令

```bash
# 训练
python scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py \
    --gpus 1 --save nwp_7day --data_dir datasets/SST-PREDICT/ \
    --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml

# 断点续训
python scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py \
    --gpus 1 --save nwp_7day --data_dir datasets/SST-PREDICT/ \
    --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml \
    --ckpt_name last.ckpt

# 测试
python scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py \
    --gpus 1 --test --save nwp_7day --data_dir datasets/SST-PREDICT/ \
    --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml \
    --ckpt_name model-epoch=XXX.ckpt

# Persistence Baseline
python scripts/cuboid_transformer/nwp_sst/persistence_baseline.py \
    --data_dir datasets/SST-PREDICT/

# 可视化
python scripts/cuboid_transformer/nwp_sst/visualize_logs.py \
    --exp_dir experiments/nwp_7day/ \
    --ckpt_path experiments/nwp_7day/checkpoints/model-epoch=XXX.ckpt \
    --data_dir datasets/SST-PREDICT/ \
    --save experiment_summary
```

---

## 18. Persistence Baseline

**脚本**: `scripts/cuboid_transformer/nwp_sst/persistence_baseline.py`

将输入序列最后一天的 SSTA 作为未来所有时刻的预测:
```python
last_ssta = X[:, -1:, :, :, 0:1]
pred = last_ssta.expand(-1, T_out, -1, -1, -1)
```

使用与 Earthformer **完全相同**的 mask、聚合和温度转换流程。输出格式对齐 `test_metrics.csv`，可直接逐行对比 RMSE/MAE，判断模型是否学得比"重复昨天"更有价值。

---

## 附录: 关键文件速查

| 文件 | 内容 |
|------|------|
| `src/earthformer/cuboid_transformer/cuboid_transformer.py` | 完整模型: CuboidAttention, Encoder, Decoder, CuboidTransformerModel |
| `src/earthformer/cuboid_transformer/cuboid_transformer_patterns.py` | 注意力模式注册表 (axial, spatial_lg, divided_st 等) |
| `src/earthformer/cuboid_transformer/spatial_frequency_branch.py` | 空间频率分支 (SpatialFrequencyBranch, ~6K参数) |
| `src/earthformer/cuboid_transformer/utils.py` | RMSNorm, padding, 位置嵌入, 初始化 |
| `src/earthformer/datasets/nw_pacific_dataset.py` | 数据加载与 Dataset 构建 (4通道, stride=3, 7天) |
| `scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py` | 训练入口 + NWPPredictionModule |
| `scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml` | 训练配置 |
| `scripts/cuboid_transformer/nwp_sst/visualize_logs.py` | 训练曲线 + 测试指标 + 预测图 (自动读cfg) |
| `scripts/cuboid_transformer/nwp_sst/persistence_baseline.py` | Persistence 基线测试 |
| `scripts/datasets/preprocess_nwp.py` | 空间裁剪+单位转换 |
| `scripts/datasets/preprocess_sla.py` | SLA 数据预处理 |
| `scripts/datasets/generate_ocean_mask.py` | 海陆掩码生成 |
| `scripts/datasets/compute_climatology.py` | 逐日气候态计算 |
| `scripts/datasets/compute_ssta.py` | SSTA 计算 |
| `scripts/datasets/inspect_sla_data.py` | SLA 数据探查工具 |
| `docs/loss_function详解.md` | 损失函数详细说明 |
