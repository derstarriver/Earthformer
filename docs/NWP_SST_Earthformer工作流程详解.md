# Earthformer 西北太平洋 SSTA 预测 — 工作流程详解

> 基于 `scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py` 和 `cfg_nwp.yaml` 配置
> 最后更新: 7通道物理感知输入, 多物理损失函数, 7天预测

---

## 目录

1. [任务定义](#1-任务定义)
2. [数据管线](#2-数据管线)
3. [模型架构总览](#3-模型架构总览)
4. [初始卷积编码](#4-初始卷积编码)
5. [位置编码](#5-位置编码)
6. [编码器](#6-编码器)
7. [Cuboid Attention 逐层策略](#7-cuboid-attention)
8. [全局向量](#8-全局向量)
9. [解码器](#9-解码器)
10. [最终上采样与投影](#10-最终上采样与投影)
11. [损失函数 — 多物理损失](#11-损失函数)
12. [评估指标](#12-评估指标)
13. [优化器与学习率调度](#13-优化器与学习率调度)
14. [完整数据形状流转表](#14-数据形状流转表)
15. [配置说明](#15-配置说明)
16. [训练流程](#16-训练流程)
17. [模型诊断套件](#17-模型诊断套件)
18. [已知问题与改进方向](#18-已知问题与改进方向)
19. [附录: 关键文件速查](#附录-关键文件速查)

---

## 1. 任务定义

| 项目 | 值 |
|------|-----|
| 输入 | 14 天 × 161×241 × **7 通道** [ssta, u10, v10, sla, grad_x, grad_y, advection] |
| 输出 | 7 天 × 161×241 × 1 通道 [ssta] |
| 区域 | 10°N–50°N, 120°E–180°E |
| 分辨率 | 0.25° |
| 数据源 | ERA5 (SST/Wind) + CMEMS AVISO (SLA), 2001-2025 日数据 |
| 模型参数量 | ~10M (scale_alpha=0.4, [1,2,2] 初始下采样) |

### 与原始 ENSO 任务的区别

| | ENSO (原) | NWP SSTA (当前) |
|------|----------|-----|
| 网格 | 24×48 | 161×241 |
| 输入长度 | 12 月 | 14 天 |
| 输出长度 | 26 月 | 7 天 |
| 输入通道 | 4 (sst, t300, ua, va) | **7** (ssta, u10, v10, sla, **grad_x, grad_y, advection**) |
| 输出通道 | 4 | 1 (仅 ssta) |
| 编码器层数 | 2 | 3 |
| 注意力策略 | 全 axial | 逐层不同 |
| 全局向量 | 关闭 | 8 个 |
| 初始下采样 | [1,1,2] | [1,4,4] |
| scale_alpha | — | 0.4 |
| 参数量 | ~7M | ~10M |
| 损失函数 | MSE | **多物理损失** (MSE + 梯度 + 时间趋势 + 反持续性) |
| 步长 | 1 | 3 (训练集) |
| 评估 | Niño 相关系数 | 逐像素 RMSE/MAE (°C)，10项诊断套件 |

---

## 2. 数据管线

### 2.1 数据来源

ERA5 (SST/Wind) + CMEMS AVISO (SLA)，经过以下预处理步骤：

| 步骤 | 脚本 | 输入 → 输出 |
|------|------|-----------|
| 空间裁剪+单位转换 | `preprocess_nwp.py` | SST.nc + wind01-22.nc → SST_cropped.nc + Wind_cropped.nc |
| 海陆掩码 | `generate_ocean_mask.py` | SST_cropped.nc → mask.npy |
| 逐日气候态 | `compute_climatology.py` | SST_cropped.nc → climatology.nc |
| SSTA 计算 | `compute_ssta.py` | SST_cropped.nc + climatology.nc → ssta.nc |
| SLA 预处理 | `preprocess_sla.py` | sladata/ → SLA_cropped.nc |

### 2.2 训练时数据加载

**文件**: `src/earthformer/datasets/nw_pacific_dataset.py`

**函数**: `build_dataloaders(data_dir, batch_size, num_workers)` → `(train, val, test, stats)`

**处理流程**：

```
1. 读取 ssta.nc + Wind_cropped.nc + SLA_cropped.nc + mask.npy
       ↓
2. 归一化 (仅用训练集 2001-2022 计算 per-channel mean/std)
   ssta = (ssta - mean) / std
   u10  = (u10  - mean) / std
   v10  = (v10  - mean) / std
   sla  = (sla  - mean) / std
       ↓
3. 掩码陆地 (归一化后) → 陆地值 ≡ 0
       ↓
4. 计算物理派生通道 (仅海洋, 训练集统计归一化)
   grad_x    = ∂(SSTA)/∂lon, z-score normalized
   grad_y    = ∂(SSTA)/∂lat, z-score normalized
   advection = -(u10·∇SSTA), z-score normalized   ← 风驱平流项
       ↓
5. 堆叠 7 通道
   data = stack([ssta, u10, v10, sla, grad_x, grad_y, advection]) → (T, 161, 241, 7)
       ↓
6. 按年切分
   训练: 2001-2022 (8035天 → ~2672样本, stride=3)
   验证: 2023-2024 (731天  → ~715样本, stride=1)
   测试: 2025        (365天  → ~349样本, stride=1)
       ↓
7. 滑窗采样 (NWPacificDataset)
   每样本: 14天输入 + 7天输出 = 21天窗口
   stride=3: 训练降低样本相关性, 防过拟合
       ↓
8. DataLoader → (B, 14, 161, 241, 7), (B, 7, 161, 241, 1), mask
```

### 2.3 物理通道的设计动机

| 通道 | 物理意义 | 解决的问题 |
|------|---------|-----------|
| `grad_x` (∂SSTA/∂lon) | 东西向温度梯度 | 帮助 attention 区分锋面和均匀区 |
| `grad_y` (∂SSTA/∂lat) | 南北向温度梯度 | 同上，保流场方向性 |
| `advection` (-u·∇SSTA) | 风驱平流趋势 | 显式注入物理动力学信号，避免纯数据驱动 |

**原理**: Attention 对 SSTA 的加权求和天然趋向平滑，显式提供梯度通道相当于给模型"指路"——高梯度区域应保持锐利。每个派生通道独立进行 z-score 归一化（用训练集海洋像素统计），确保与基础 4 通道尺度一致。

**标准化统计量**: 保存为 `normalization_stats.npz`，包含 per-channel mean/std。旧版 4 通道统计自动兼容，新通道（grad_x/grad_y/advection）的统计在 `build_data_array` 中按需计算。

---

## 3. 模型架构总览

```
输入 (B, 14, 161, 241, 7)
   │
   ▼
┌──────────────────────────────────────┐
│ InitialEncoder                       │
│  Conv2D×3 → PatchMerging3D(1,4,4)   │
└──────────────────────────────────────┘
   │ (B, 14, 41, 61, 64)
   ▼
┌──────────────────────────────────────┐
│ Encoder PosEmbed (t+h+w)             │
└──────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│ Encoder Block 0  [axial, dim=64]     │
│  2 × CuboidSelfAttentionLayer        │
│    + PositionwiseFFN                 │
└──────────────────────────────────────┘
   │ (B, 14, 41, 61, 64)  → mem[0]
   ▼ PatchMerge(1,2,2)
   │ (B, 14, 21, 31, ~80)
   ▼
┌──────────────────────────────────────┐
│ Encoder Block 1  [spatial_lg_8, dim≈80] │
│  2 × CuboidSelfAttentionLayer        │
└──────────────────────────────────────┘
   │ (B, 14, 21, 31, ~80)  → mem[1]
   ▼ PatchMerge(1,2,2)
   │ (B, 14, 11, 16, ~97)
   ▼
┌──────────────────────────────────────┐
│ Encoder Block 2  [divided_st, dim≈97] │
│  2 × CuboidSelfAttentionLayer        │
└──────────────────────────────────────┘
   │ (B, 14, 11, 16, ~97)  → mem[2]
   │
   │  多尺度记忆: mem[0], mem[1], mem[2]
   │  全局向量: 8 × 64维, 逐层传播
   │
   ▼
┌──────────────────────────────────────┐
│ Decoder Init (z_init=zeros)          │
│  zeros(7, 11, 16, ~97) + PosEmbed    │
└──────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│ Decoder Block 2 → cross(mem[2])      │
└──────────────────────────────────────┘
   │ Upsample
   ▼
┌──────────────────────────────────────┐
│ Decoder Block 1 → cross(mem[1])      │
└──────────────────────────────────────┘
   │ Upsample
   ▼
┌──────────────────────────────────────┐
│ Decoder Block 0 → cross(mem[0])      │
└──────────────────────────────────────┘
   │ (B, 7, 41, 61, 64)
   ▼
┌──────────────────────────────────────┐
│ FinalDecoder                         │
│  Upsample → Conv2D×2                 │
└──────────────────────────────────────┘
   │ (B, 7, 161, 241, 64)
   ▼
┌──────────────────────────────────────┐
│ dec_final_proj: Linear(64, 1)        │
└──────────────────────────────────────┘
   │
   ▼
输出: ΔSSTA (B, 7, 161, 241, 1) → training_step: X_last + Δ = absolute SST
```

---

## 4. 初始卷积编码

**配置**:
```yaml
initial_downsample_type: "conv"
initial_downsample_scale: [1, 4, 4]      # H:4×, W:4×, T:不变
initial_downsample_conv_layers: 3
initial_downsample_activation: "leaky"
```

**流程**:
```
输入: (B, 14, 161, 241, 7)

Step 1: 展平时空维度
    reshape → (B×14, 161, 241, 7)
    permute → (B×14, 7, 161, 241)

Step 2: Conv2D×3 (3×3, GroupNorm+LeakyReLU)
    Conv0: 7 → 64
    Conv1: 64 → 64
    Conv2: 64 → 64

Step 3: 恢复
    permute + reshape → (B, 14, 161, 241, 64)

Step 4: PatchMerging3D(1,4,4)
    H: 161→41, W: 241→61

输出: (B, 14, 41, 61, 64)
```

---

## 5. 位置编码

**配置**: `pos_embed_type: "t+h+w"`

三个独立的可学习 Embedding 表:
```python
T_embed: nn.Embedding(maxT=14, embed_dim=dim)
H_embed: nn.Embedding(maxH=H,  embed_dim=dim)
W_embed: nn.Embedding(maxW=W,  embed_dim=dim)

x = x + T_embed + H_embed + W_embed
```

Encoder 中加一次；Decoder 各层上采样后重加 (`dec_hierarchical_pos_embed: true`)。

---

## 6. 编码器

**源码**: `CuboidTransformerEncoder`, `cuboid_transformer.py`

**配置**:
```yaml
enc_depth: [2, 2, 2]    # 3层, 每层2个attention block
downsample: 2           # 块间 PatchMerge (1,2,2)
base_units: 64
scale_alpha: 0.4        # 通道增长减速, 控制参数量
```

**通道数** (`scale_alpha: 0.4`):
```
Block 0: int(64 × 2^(0×0.4)) = 64
Block 1: int(64 × 2^(1×0.4)) ≈ 80
Block 2: int(64 × 2^(2×0.4)) ≈ 97
```

**分辨率变化** (`initial_downsample_scale: [1,4,4]`):
```
输入编码器: (14, 41, 61, 64)
  Block 0 (axial):         (14, 41, 61, 64)   → mem[0]
  PatchMerge(1,2,2):       (14, 21, 31, ~80)
  Block 1 (spatial_lg_8):  (14, 21, 31, ~80)   → mem[1]
  PatchMerge(1,2,2):       (14, 11, 16, ~97)
  Block 2 (divided_st):    (14, 11, 16, ~97)   → mem[2]
```

**多尺度记忆**: 每层输出保存供解码器跨注意力使用:
```
mem[0]: (B, 14, 41, 61, 64)   ← 高分辨率, 浅语义 → Dec Block 0
mem[1]: (B, 14, 21, 31, ~80)  ← 中分辨率         → Dec Block 1
mem[2]: (B, 14, 11, 16, ~97)  ← 低分辨率, 深语义 → Dec Block 2
```

---

## 7. Cuboid Attention — 逐层策略

**配置**:
```yaml
self_pattern:       ["axial", "spatial_lg_8", "divided_st"]
cross_self_pattern: ["axial", "spatial_lg_8", "divided_st"]
cross_pattern:      ["cross_1x1", "cross_1x1", "cross_1x1"]
```

### 7.1 Block 0: axial
- T-full (14, 1, 1) + H-full (1, H, 1) + W-full (1, 1, W)
- 最浅层最大网格，轴向分解最省计算

### 7.2 Block 1: spatial_lg_8
- T-full (14, 1, 1) + local 8×8 + dilated 8×8
- 局部保纹理，膨胀扩大感受野

### 7.3 Block 2: divided_st
- T-full (14, 1, 1) + full spatial (1, H, W)
- 最深最小分辨率，全空间 attention token 数可控

### 7.4 跨注意力: cross_1x1 (全三层)
```
cuboid_hw = (1, 1)      → 逐像素跨注意力
Q 来自 Decoder (7天), K/V 来自 Encoder memory (14天)
```

---

## 8. 全局向量

**配置**:
```yaml
num_global_vectors: 8
use_dec_self_global: true
dec_self_update_global: true
use_dec_cross_global: true
use_global_vector_ffn: true
use_global_self_attn: false
```

8 个可学习向量在各 cuboid 之间传递长距离信息。Cuboid 内部做 local attention，跨 cuboid 通过这 8 个向量中转。

---

## 9. 解码器

**配置**:
```yaml
dec_depth: [2, 2, 2]
dec_use_first_self_attn: false    # Block 2 先cross后self
dec_hierarchical_pos_embed: true  # 上采样后重加位置编码
```

解码器从零向量 `zeros(7, H, W, dim)` + 位置编码初始化，3 层逐级上采样（nearest + Conv2D），每层 cross-attend 对应 encoder memory。

---

## 10. 最终上采样与投影

```
(B, 7, 41, 61, 64)
  → Upsample3DLayer → (B, 7, 161, 241, 64)
  → Conv2D×2 (GroupNorm+LeakyReLU)
  → dec_final_proj: Linear(64, 1) → (B, 7, 161, 241, 1)
```

---

## 11. 损失函数 — 多物理损失

**当前实现** (`_compute_loss` in `train_nwp_sst.py`):

```python
pred = X_last + delta_pred   # Δ → 绝对 SSTA

# 1. 逐天加权 MSE (基础损失)
day_w = linspace(0.8, 1.5, 7)       # 远期预测权重更高，抵销 persistence drift
loss_mse = MSE(pred, Y, weight=day_w)

# 2. 空间梯度 MSE（在纯 delta 上计算）
delta = pred - X_last
delta_ref = Y - X_last
loss_grad = MSE(grad(delta), grad(delta_ref))   # 逼迫 delta 具有正确的空间结构

# 3. 时间趋势 MSE（逐日变化量）
dt_pred = pred[t+1] - pred[t]
dt_true = Y[t+1] - Y[t]
loss_tend = MSE(dt_pred, dt_true)    # main term preventing persistence shortcut

# 4. 反持续性惩罚
loss_anti = relu(loss_mse_plain - 0.95 × persist_mse)  # 模型不能比 persistence 差

total = loss_mse + 0.5 × loss_grad + 5.0 × loss_tend + 0.3 × loss_anti
```

### 设计动机

| 损失项 | 权重 | 解决的核心问题 |
|--------|------|--------------|
| `loss_mse` | 1.0 | 基础预测精度 |
| `loss_grad` | 0.5 | 锋面抹平 — 惩罚 delta 的空间梯度方向错误 |
| `loss_tend` | **5.0** | 时间复制 — 强制每日变化量匹配真实动力学 |
| `loss_anti` | 0.3 | 退化防护 — 确保模型至少不比 persistence 差 |

### 关键设计决策

- **梯度损失在 delta 上计算而非绝对 SSTA**: `X_last` 的空间梯度主导 `pred` 的梯度，会导致 loss_grad 变成常数。在纯 `delta` 上计算使梯度信号聚焦于模型实际预测的变化场空间结构。
- **趋势损失用 MSE 而非 Pearson 相关**: 早期版本使用 `_masked_pearson`，但打平 `B×(T-1)×H×W` 后计算的 Pearson 实际上是测量空间模式相关 (≈0.98)，而不是时间动态相关 (≈0.09)。纯 MSE 直接惩罚 `dt_pred ≈ 0` 这种 persistence-like 输出。
- **λ_tend = 5.0**: 趋势项必须为主导项，否则模型会找到 `delta[t] ≈ delta[t±1]` 的局部极小。

### 训练监控日志

除 `train_loss` 外，额外 log 四个分量：
```
train_mse, train_grad, train_tend, train_anti
```
用于判断哪个物理约束在主导训练。正常训练初期 `train_tend` 应在 0.08-0.15 间，随训练下降。

---

## 12. 评估指标

### 12.1 训练/验证 (归一化空间)

```python
self.valid_mse(pred_ocean, Y_ocean)   # torchmetrics
self.valid_mae(pred_ocean, Y_ocean)
self.log('valid_loss', plain_mse)     # 纯 MSE (无物理分量) 用于 checkpoint 选择
```

Checkpoint 选择依据: `valid_mse_epoch` (min mode)。

### 12.2 测试 (分天, °C 转换)

```python
# 累计每天的平方误差
sq_err = ((pred - Y)² × mask).sum(dim=(0,2,3,4))   # (7,)
abs_err = ((pred - Y).abs() × mask).sum(dim=(0,2,3,4))

# 全局归一化
n_total = Σ mask.sum() over batches
mse_per_day = sq_err / n_total

# 转摄氏度
rmse_degC = sqrt(mse_per_day × ssta_std²)
mae_degC  = mae_per_day × ssta_std
```

输出示例:
```
  Day   MSE( norm )    MAE( norm )   RMSE(°C)    MAE(°C)
    1     0.xxxxxx       0.xxxxxx      0.xxxx       0.xxxx
   ...
    7     0.xxxxxx       0.xxxxxx      0.xxxx       0.xxxx
  avg     0.xxxxxx       0.xxxxxx      0.xxxx       0.xxxx
```

结果保存至 `test_metrics.csv`。

---

## 13. 优化器与学习率调度

### AdamW
```
lr = 1e-4, weight_decay = 1e-4

参数分组:
  Group 1 (含 weight_decay): 非 LayerNorm, 非 bias
  Group 2 (无 weight_decay): LayerNorm + bias
```

### 两阶段调度
```
Warmup (10%):  lr 0 → 1e-4 (线性)
Cosine (90%):  lr 1e-4 → 1e-4 × min_lr_ratio (余弦衰减)
```

### 正则化
```yaml
attn_drop: 0.2
proj_drop: 0.2
ffn_drop: 0.3      # FFN 参数量大, 用更高 dropout
wd: 1e-4
gradient_clip_val: 1.0
early_stop_patience: 30
max_epochs: 100
```

---

## 14. 完整数据形状流转表

> 基于 `scale_alpha: 0.4`, `initial_downsample_scale: [1,4,4]`

| 阶段 | 形状 | 说明 |
|------|------|------|
| 原始 SSTA | (9131, 161, 241) | 异常值 °C |
| 原始 Wind | (9131, 161, 241, 2) | u10, v10 m/s |
| 原始 SLA | (9131, 161, 241) | CMEMS AVISO, m |
| 合并+Base归一化 | (9131, 161, 241, 4) | ssta, u10, v10, sla z-score |
| +物理通道归一化 | (9131, 161, 241, 7) | +grad_x, grad_y, advection |
| 训练数据段 | (8035, 161, 241, 7) | 2001-2022 |
| 验证数据段 | (731, 161, 241, 7) | 2023-2024 |
| 测试数据段 | (365, 161, 241, 7) | 2025 |
| 单样本 | (21, 161, 241, 7) | 14输入+7输出 |
| 模型输入 X | (B, 14, 161, 241, 7) | |
| 模型输出 Y_true | (B, 7, 161, 241, 1) | 仅 SSTA 的 absolute 值 |
| InitialEncoder 后 | (B, 14, 41, 61, 64) | 4×4 下采样 |
| Enc Block 0 后 | (B, 14, 41, 61, 64) | mem[0] |
| PatchMerge 后 | (B, 14, 21, 31, ~80) | 2×2 |
| Enc Block 1 后 | (B, 14, 21, 31, ~80) | mem[1] |
| PatchMerge 后 | (B, 14, 11, 16, ~97) | 2×2 |
| Enc Block 2 后 | (B, 14, 11, 16, ~97) | mem[2] |
| Decoder 初始化 | (B, 7, 11, 16, ~97) | 零向量+位置编码 |
| Dec Block 2 后 | (B, 7, 11, 16, ~97) | |
| Upsample 后 | (B, 7, 21, 31, ~80) | |
| Dec Block 1 后 | (B, 7, 21, 31, ~80) | |
| Upsample 后 | (B, 7, 41, 61, 64) | |
| Dec Block 0 后 | (B, 7, 41, 61, 64) | |
| FinalDecoder 后 | (B, 7, 161, 241, 64) | 恢复原始分辨率 |
| 最终投影 ΔSST | (B, 7, 161, 241, 1) | |
| → X_last + Δ | (B, 7, 161, 241, 1) | training_step 中转换 |
| 掩码 | (B, 161, 241, 1) | 1=海洋, 0=陆地 |

---

## 15. 配置说明

**文件**: `scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml`

### 关键参数速查

| 分类 | 参数 | 值 | 说明 |
|------|------|-----|------|
| **数据** | `data_dir` | `datasets/SST-PREDICT/` | |
| | `in_len` | 14 | 输入天数 |
| | `out_len` | 7 | 输出天数 |
| | `var_names` | `[ssta, u10, v10, sla, grad_x, grad_y, advection]` | **7通道** |
| **模型** | `data_channels` | **7** | 输入通道数 |
| | `input_shape` | `[14, 161, 241, 7]` | |
| | `target_shape` | `[7, 161, 241, 1]` | |
| | `base_units` | 64 | 基础通道数 |
| | `scale_alpha` | 0.4 | 通道增长减速 |
| | `enc_depth` | `[2,2,2]` | 3层, 每层2块 |
| | `dec_depth` | `[2,2,2]` | 同上 |
| | `num_global_vectors` | 8 | 全局向量数 |
| | `initial_downsample_scale` | `[1,4,4]` | 空间16×压缩 |
| | `num_heads` | 4 | 注意力头数 |
| **注意力** | `self_pattern` | `["axial","spatial_lg_8","divided_st"]` | |
| | `cross_pattern` | `["cross_1x1"]×3` | |
| **正则化** | `attn_drop` | 0.2 | |
| | `proj_drop` | 0.2 | |
| | `ffn_drop` | 0.3 | |
| | `wd` | 1e-4 | |
| **优化** | `lr` | 1e-4 | |
| | `total_batch_size` | 16 | 有效batch |
| | `micro_batch_size` | 2 | |
| | `max_epochs` | 100 | |
| | `early_stop_patience` | 30 | |

### 损失函数参数（硬编码在 `_compute_loss` 中）

| 参数 | 值 | 说明 |
|------|-----|------|
| `day_weight_range` | [0.8, 1.5] | 逐天 MSE 线性权重 |
| `λ_grad` | 0.5 | 梯度损失系数 |
| `λ_tend` | 5.0 | 趋势损失系数 (主导项) |
| `λ_anti` | 0.3 | 反持续性损失系数 |
| `persist_ratio` | 0.95 | 反持续性触发阈值 |

---

## 16. 训练流程

### 16.1 实验目录结构

```
experiments/nwp_7day/
├── hparams.json          ← 超参数 (JSON)
├── metrics.csv           ← 每轮指标 (CSV, 断点续训追加不覆盖)
│   列: epoch, train_loss, valid_loss, valid_mse, valid_mae, learning_rate
├── test_metrics.csv      ← 测试分天指标 (°C, 7天)
├── cfg.yaml              ← 配置文件备份
├── diagnosis/            ← 诊断套件输出 (10张图 + 终端报告)
└── checkpoints/
    ├── model-epoch=xxx.ckpt   ← 最优N个
    ├── last.ckpt              ← 最新 (断点续训)
    └── best_model.pt          ← 纯权重 (推理用)
```

### 16.2 运行命令

```bash
# 训练 (从头)
python scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py \
    --gpus 1 --save nwp_7day --data_dir datasets/SST-PREDICT/ \
    --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml

# 断点续训 (保留已有 metrics)
python scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py \
    --gpus 1 --save nwp_7day --data_dir datasets/SST-PREDICT/ \
    --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml \
    --ckpt_name last.ckpt

# 测试 (指定 checkpoint)
python scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py \
    --gpus 1 --test --save nwp_7day --data_dir datasets/SST-PREDICT/ \
    --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml \
    --ckpt_name model-epoch=XXX.ckpt

# 模型诊断
python scripts/cuboid_transformer/nwp_sst/diagnose_model.py \
    --exp_dir experiments/nwp_7day/ \
    --ckpt_name model-epoch=XXX.ckpt \
    --data_dir datasets/SST-PREDICT/ \
    --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml
```

---

## 17. 模型诊断套件

**文件**: `scripts/cuboid_transformer/nwp_sst/diagnose_model.py`

修复了早期 stats 加载 bug（`build_dataloaders` 返回 dict 无 `.files` 属性导致 stats={}）和 `collect_predictions` 缺失 `X_last + delta` 转换。

### 三大类诊断

**I. 空间诊断 (1.1-1.3)**
| 实验 | 内容 | 关键指标 |
|------|------|---------|
| 1.1 | 逐像素 RMSE 热力图 | Mean RMSE, top-5% 误差集中度 |
| 1.2 | 区域 RMSE (6 个海洋学区域) | Kuroshio, Oyashio, 赤道等分区 |
| 1.3 | 梯度误差 | grad bias (<0=过平滑), grad angle error (>45°=方向错误) |

**II. 时间诊断 (2.1-2.4)**
| 实验 | 内容 | 关键指标 |
|------|------|---------|
| 2.1 | RMSE 逐日增长 | slope ratio vs persistence |
| 2.2 | 时间自相关衰减 | ACF lag-6 pred (应接近 truth 0.74, 而非 0.95+) |
| 2.3 | 逐像素 vs Persistence | 负改进像素比例 |
| 2.4 | 逐像素时间相关 | Day 7 低相关 (<0.5) 像素比例 |

**III. 物理诊断 (3.1-3.3)**
| 实验 | 内容 | 关键指标 |
|------|------|---------|
| 3.1 | SST 趋势 | dSST/dt 整体相关, 分 bin RMSE |
| 3.2 | 平流一致性 | cos(dSST/dt, -u·∇SST) |
| 3.3 | 扰动传播 | 注入 +1°C, 追踪质量和重心移动 |

### 决策树输出
根据诊断指标自动标注瓶颈: PERSISTENCE_LIKE, NO_TENDENCY_SKILL, IMAGE_FITTER 等。

---

## 18. 已知问题与改进方向

### 当前状态 (2026-07 基线)

| 指标 | 数值 | 评估 |
|------|------|------|
| avg RMSE | 0.483°C | 可接受, 略优于 persistence |
| grad bias | -0.013 | ✅ 空间结构保持良好 |
| grad angle error | 49.7° | ✅ 梯度方向基本正确 |
| persistence worse % | 6.9% | ✅ 仅少数像素差于 persistence |
| ACF lag-6 pred | 0.985 | ❌ 输出几乎不随时间变化 |
| dSST/dt corr | 0.085 | ❌ 无法学习时间动态 |
| RMSE growth ratio | 0.93 | ❌ 误差增速接近 persistence |

### 根本原因

模型一次性同时输出 7 步预测（Decoder query 之间无因果依赖），可以找到"所有帧输出相近值"的低 MSE 解。损失函数（即使 λ_tend=5.0）的梯度信号不足以在一次性 7 帧输出的架构下打破此行为。

### 已验证无效的方向
- FFT / spectral branch: 频域问题 ≠ SST 动力学问题
- Pearson 相关作为趋势损失: 打平空间维度后测的是空间模式相关 (≈0.98)，不是时间动态 (≈0.09)

### 推荐改进路线
1. **自回归训练**: 模型只预测 1 天，预测结果滚入输入窗口循环 7 次
2. **因果 Decoder**: 给 Decoder 加 causal temporal attention
3. **物理先验 + 残差**: 用简单平流方程做基线，模型只学残差项

---

## 附录: 关键文件速查

| 文件 | 内容 |
|------|------|
| `src/earthformer/cuboid_transformer/cuboid_transformer.py` | 完整模型: CuboidAttention, Encoder, Decoder, CuboidTransformerModel |
| `src/earthformer/cuboid_transformer/cuboid_transformer_patterns.py` | 注意力模式注册表 |
| `src/earthformer/cuboid_transformer/utils.py` | RMSNorm, 位置嵌入, 初始化 |
| `src/earthformer/datasets/nw_pacific_dataset.py` | 数据加载 (7通道, 物理派生通道, stride=3) |
| `scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py` | 训练入口 + NWPPredictionModule + 多物理损失 |
| `scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml` | 训练配置 |
| `scripts/cuboid_transformer/nwp_sst/diagnose_model.py` | 10项缺陷诊断套件 |
| `scripts/cuboid_transformer/nwp_sst/persistence_baseline.py` | Persistence 基线 |
| `scripts/datasets/preprocess_nwp.py` | SST/Wind 预处理 |
| `scripts/datasets/preprocess_sla.py` | SLA 预处理 |
| `scripts/datasets/generate_ocean_mask.py` | 海陆掩码 |
| `scripts/datasets/compute_climatology.py` | 气候态计算 |
| `scripts/datasets/compute_ssta.py` | SSTA 计算 |
