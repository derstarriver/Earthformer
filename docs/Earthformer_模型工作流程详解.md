# Earthformer ENSO/SST 预测模型工作流程详解

> 基于 `scripts/cuboid_transformer/enso/train_cuboid_enso.py` 的配置，以 `cfg.yaml` 默认参数为准。

---

## 目录

1. [模型总览](#1-模型总览)
2. [数据管线](#2-数据管线)
3. [初始卷积编码 (InitialEncoder)](#3-初始卷积编码)
4. [位置编码 (PosEmbed)](#4-位置编码)
5. [编码器 (CuboidTransformerEncoder)](#5-编码器)
6. [Cuboid Attention 核心机制](#6-cuboid-attention)
7. [解码器 (CuboidTransformerDecoder)](#7-解码器)
8. [最终上采样解码 (FinalDecoder)](#8-最终上采样解码)
9. [损失函数与评估指标](#9-损失函数与评估指标)
10. [优化器与学习率调度](#10-优化器与学习率调度)
11. [完整数据形状流转表](#11-完整数据形状流转表)
12. [可用的注意力策略清单](#12-可用的注意力策略清单)

---

## 1. 模型总览

```
输入 (B, 12, 24, 48, 4)
   │  12个月 × 24lat × 48lon × 4通道(sst,t300,ua,va)
   │
   ▼
┌─────────────────────────────────────┐
│ InitialEncoder                      │
│  Conv2D×2 → PatchMerging3D(1,1,2)   │  ← 经度减半(48→24), 通道 4→64
└─────────────────────────────────────┘
   │ (B, 12, 24, 24, 64)
   ▼
┌─────────────────────────────────────┐
│ Encoder PosEmbed (t+h+w)            │  ← 可学习时间+空间位置嵌入
└─────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────┐
│ Encoder Block 1                     │
│  StackCuboidSelfAttentionBlock      │  ← axial attention, dim=64
│  depth=1, 1 attn + 1 FFN            │
└─────────────────────────────────────┘
   │ (B, 12, 24, 24, 64)  → mem_l[0]
   ▼
┌─────────────────────────────────────┐
│ PatchMerging3D(1,2,2)               │  ← H/W各减半, 通道 64→128
└─────────────────────────────────────┘
   │ (B, 12, 12, 12, 128)
   ▼
┌─────────────────────────────────────┐
│ Encoder Block 2                     │
│  StackCuboidSelfAttentionBlock      │  ← axial attention, dim=128
│  depth=1                            │
└─────────────────────────────────────┘
   │ (B, 12, 12, 12, 128)  → mem_l[1]
   │
   │   多尺度记忆 mem_l = [mem_l[0], mem_l[1]]
   │
   ▼
┌─────────────────────────────────────┐
│ Decoder Init (z_init_method=zeros)   │
│  zeros(26, 12, 12, 128) + PosEmbed  │  ← 26步零初始化+位置编码
│  → Linear(128,128)                  │
└─────────────────────────────────────┘
   │ (B, 26, 12, 12, 128)
   ▼
┌─────────────────────────────────────┐
│ Decoder Block 2 (最深层)             │
│  dec_use_first_self_attn=False       │  ← 先cross-attn到mem_l[1]
│  cross_attn: cross_1x1               │
└─────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────┐
│ Upsample3DLayer                     │  ← 最近邻上采样+Conv2D
│  (26,12,12)→(26,24,24), 128→64      │
└─────────────────────────────────────┘
   │ (B, 26, 24, 24, 64)
   ▼
┌─────────────────────────────────────┐
│ Decoder Block 1 (最浅层)             │
│  self_attn: axial                   │
│  cross_attn: cross_1x1 → mem_l[0]   │
└─────────────────────────────────────┘
   │ (B, 26, 24, 24, 64)
   ▼
┌─────────────────────────────────────┐
│ FinalDecoder                        │
│  Upsample3DLayer → Conv2D×1         │  ← 恢复经度 24→48
└─────────────────────────────────────┘
   │ (B, 26, 24, 48, 64)
   ▼
┌─────────────────────────────────────┐
│ dec_final_proj: Linear(64, 4)       │  ← 逐像素投影到4通道
└─────────────────────────────────────┘
   │
   ▼
输出 (B, 26, 24, 48, 4)
```

**模型参数量**: 约 1.4M，显存占用约 5.6 MB

---

## 2. 数据管线

### 2.1 数据来源

| 数据 | 文件 | 原始形状 | 用途 |
|------|------|---------|------|
| CMIP6 | CMIP_train.nc | (2265, 36, 24, 72) | 训练 (前2265行) |
| CMIP5 | CMIP_train.nc | (2380, 36, 24, 72) | 训练 (后2380行) |
| SODA | SODA_train.nc | (100, 36, 24, 72) | 验证+测试 |

每个文件包含 4 个变量: `sst`, `t300`, `ua`, `va`。每行 `month=36` 代表 3 个日历年 (12月/年 × 3)。

### 2.2 预处理 (preprocess_enso.py, 一次执行)

**步骤 1: 经度过滤**
```
CMIP_train.nc (4645, 36, 24, 72)
    → 过滤 lon ∈ [95°E, 330°E]
    → (4645, 36, 24, 48)
```

**步骤 2: data_transform (每个变量)**

```python
# 以 CMIP6 为例 (前 2265 行)
data_transform(vals[:2265], num_years_per_model=151)

# 内部:
# 1. 按模型拆分: 2265 ÷ 151 = 15 个 CMIP6 模式
# 2. np.stack → (151, 36, lat, lon, 15)
# 3. fold(size=36, stride=12): 展开重叠的年份窗口
#    151*12 + 24 = 1836 个月/模型
# 4. cat_over_last_dim: 15*1836 = 27540 个月
```

**fold 函数原理:**
```
原始行结构 (每行36个月, 相邻行重叠24个月):
Row 0:  月0 ────────── 月35
Row 1:      月12 ────────── 月47
Row 2:           月24 ────────── 月59
...

fold取每隔3行 (data[::3]):
Row 0: 月0..月35
Row 3: 月36..月71   ← 完美衔接, 无断点
Row 6: 月72..月107
...
```

**步骤 3: 通道堆叠 + 归一化**

4个变量各自完成 data_transform 后堆叠为最后一维:
```
sst:  (27540, 24, 48)  ─┐
t300: (27540, 24, 48)   ├─ stack → (27540, 24, 48, 4)
ua:   (27540, 24, 48)   │
va:   (27540, 24, 48)  ─┘

然后 min-max 归一化存入缓存 .npz
```

**步骤 4: CMIP+SODA 汇总**

| 缓存数据 | 形状 | 样本量(月) |
|---------|------|----------|
| CMIP6 | (27540, 24, 48, 4) | 27,540 |
| CMIP5 | (28968, 24, 48, 4) | 28,968 |
| SODA | (1224, 24, 48, 4) | 1,224 |

### 2.3 训练时数据加载

**滑窗构造 (prepare_inputs_targets):**

```python
in_len=12, out_len=26, samples_gap=1

# 每条样本取 12+26=38 个连续月
# samples_gap=1: 每个月滑1步
# CMIP: 56508个月 → 约 56,470 个训练样本
```

**Dataset.__getitem__ 返回值:**

```
x: (38, 24, 48, 4)    # 前12=输入, 后26=GT目标
y: (24,)               # Niño 3.4指数 (不是26而是24, 原因见§9)
```

**训练/验证/测试划分:**

```
训练: CMIP6 + CMIP5 全部 (56,508月)
验证: SODA 前50% (~610样本)
测试: SODA 后50% (~610样本)
```

---

## 3. 初始卷积编码 (InitialEncoder)

**代码位置**: `cuboid_transformer.py:2442-2520`

**配置参数:**
```yaml
initial_downsample_type: "conv"
initial_downsample_scale: [1, 1, 2]     # 仅经度方向2倍下采样
initial_downsample_conv_layers: 2       # 2层Conv2D
initial_downsample_activation: "leaky"
```

**详细流程:**

```
输入: (B, 12, 24, 48, 4)

Step 1: 展平时间维度
    reshape → (B*12, 24, 48, 4)
    permute → (B*12, 4, 24, 48)   # 转为 (N, C, H, W) 格式

Step 2: Conv2D 层×2 (K=3, S=1, P=1, GroupNorm+LeakyReLU)
    Conv0: 4 → 64, GroupNorm(1), LeakyReLU
    Conv1: 64 → 64, GroupNorm(1), LeakyReLU
    # 空间尺寸不变 (padding='same')

Step 3: 恢复时空格式
    permute → (B*12, 24, 48, 64)
    reshape → (B, 12, 24, 48, 64)

Step 4: PatchMerging3D (downsample=(1,1,2))
    # 时间不降采样(dt=1), 高度不降采样(dh=1), 宽度降采样2倍(dw=2)
    # 经度 48 → 24
    # 通道 64 → 128 (dt*dh*dw*C = 1*1*2*64 = 128)

    # 但 out_dim 默认 = max(downsample) * dim = 2 * 64 = 128
    # 实际输出通道由 Linear(128, 64) 压缩回 64

输出: (B, 12, 24, 24, 64)
```

---

## 4. 位置编码 (PosEmbed)

**代码位置**: `cuboid_transformer.py:22-94`

**配置**: `pos_embed_type: "t+h+w"`

```
T_embed: nn.Embedding(maxT=12, embed_dim=64)    # 时间嵌入
H_embed: nn.Embedding(maxH=24, embed_dim=64)    # 高度嵌入
W_embed: nn.Embedding(maxW=24, embed_dim=64)    # 宽度嵌入

前向传播:
    t_idx = [0, 1, ..., 11]      # 各时间步索引
    h_idx = [0, 1, ..., 23]      # 各纬度索引
    w_idx = [0, 1, ..., 23]      # 各经度索引

    x = x + T_embed(t_idx).reshape(12, 1, 1, 64)
          + H_embed(h_idx).reshape(1, 24, 1, 64)
          + W_embed(w_idx).reshape(1, 1, 24, 64)
```

三种位置嵌入在相同维度上相加，每个时空位置有唯一的编码。

---

## 5. 编码器 (CuboidTransformerEncoder)

**代码位置**: `cuboid_transformer.py:1649-1923`

**架构**: 2级层次结构 + PatchMerging3D 下采样

```
输入 (B, 12, 24, 24, 64)
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Encoder Block 1 (dim=64, depth=1)              │
│  ┌───────────────────────────────────────────┐   │
│  │  CuboidSelfAttentionLayer #1              │   │
│  │    cuboid_size=(12,1,1) → temporal attn  │   │  axial 模式
│  │    cuboid_size=(1,24,1)  → height attn   │   │  = 3个独立
│  │    cuboid_size=(1,1,24)  → width attn    │   │   注意力层
│  │  └→ PreNorm → reorder → QKV → attn →     │   │
│  │     relative_bias → softmax → proj       │   │
│  ├───────────────────────────────────────────┤   │
│  │  PositionwiseFFN (hidden=256)            │   │  gelu激活
│  │    LayerNorm → Linear(64→256) → gelu     │   │
│  │    → Dropout → Linear(256→64) → Dropout  │   │
│  └───────────────────────────────────────────┘   │
│  输出: (B, 12, 24, 24, 64) → 存入 mem_l[0]      │
└─────────────────────────────────────────────────┘
    │
    ▼ PatchMerging3D(1,2,2)
    │   H: 24→12, W: 24→12
    │   C: 64×4=256 → Linear→128 = 128
    │   输出: (B, 12, 12, 12, 128)
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Encoder Block 2 (dim=128, depth=1)             │
│  ┌───────────────────────────────────────────┐   │
│  │  CuboidSelfAttentionLayer #1              │   │
│  │    cuboid_size=(12,1,1) → temporal attn  │   │  axial 模式
│  │    cuboid_size=(1,12,1) → height attn    │   │
│  │    cuboid_size=(1,1,12) → width attn     │   │
│  ├───────────────────────────────────────────┤   │
│  │  PositionwiseFFN (hidden=512)            │   │
│  └───────────────────────────────────────────┘   │
│  输出: (B, 12, 12, 12, 128) → 存入 mem_l[1]      │
└─────────────────────────────────────────────────┘

编码器输出:
    mem_l[0]: (B, 12, 24, 24, 64)   ← 高分辨率、浅通道
    mem_l[1]: (B, 12, 12, 12, 128)  ← 低分辨率、深通道
```

**设计要点:**
- 2层层次结构: 每层分辨率减半、通道翻倍
- axial 注意力: 分解为时间、高度、宽度3个独立维度的 self-attention
- 残差连接: 每个 attention 和 FFN 后都有 `x = x + sublayer(x)`
- PreNorm: 归一化在子层之前

---

## 6. Cuboid Attention 核心机制

**代码位置**: `cuboid_transformer.py:386-962`

### 6.1 核心思想

将 3D 时空张量 (T×H×W) 分解为不重叠的小长方体 (cuboids)，在每个长方体内并行做 self-attention。

### 6.2 分解策略

**local (`'l'`)** : 连续元素归为一组
```
轴大小为12, cuboid=4: [0 1 2 3] [4 5 6 7] [8 9 10 11]
```

**dilated (`'d'`)** : 跨步采样
```
轴大小为12, cuboid=4: [0 3 6 9] [1 4 7 10] [2 5 8 11]
```

### 6.3 当前配置的 axial 注意力

```
第1层 (dim=64):
    cuboid #1: size=(12, 1, 1), strategy=(l,l,l), shift=(0,0,0)
        → 沿时间轴, 每个 (T,1,1) 方块内完整的12步时间注意力
    cuboid #2: size=(1, 24, 1), strategy=(l,l,l), shift=(0,0,0)
        → 沿高度轴, 逐行24个像元的空间注意力
    cuboid #3: size=(1, 1, 24), strategy=(l,l,l), shift=(0,0,0)
        → 沿宽度轴, 逐列24个像元的空间注意力

第2层 (dim=128):
    同理, HW变为12×12
```

### 6.4 单层注意力前向传播 (CuboidSelfAttentionLayer)

```
输入 x: (B, T, H, W, C)

1. Pre-norm: x = LayerNorm(x)

2. 填充: 使各维度能被 cuboid_size 整除

3. 循环移位 (shift > 0 时):
    x = torch.roll(x, shifts=(-sT, -sH, -sW), dims=(1,2,3))

4. 重排为长方体: cuboid_reorder → (B, num_cuboids, vol, C)

5. QKV投影:
    qkv = Linear(C, 3*C)(x) → (B, n_cub, vol, 3*C)
    拆分为 Q, K, V → (B, n_heads, n_cub, vol, head_dim)

6. 注意力分数:
    attn = Q @ K^T / sqrt(head_dim)  → (B, n_heads, n_cub, vol, vol)

7. 相对位置偏置 (use_relative_pos=True):
    bias = relative_bias_table[relative_pos_index]
    attn = attn + bias

8. Masked Softmax + Dropout(0.1)

9. 加权求和: out = attn @ V

10. 输出投影: proj(out) + dropout

11. 逆重排: cuboid_reorder_reverse → (B, T, H, W, C)

12. 逆循环移位

13. 去填充
```

**计算复杂度**: O(T×H×W × bT×bH×bW)
vs 全注意力: O(T²×H²×W²)

### 6.5 相对位置偏置

```
相对位置表:
    (2*bT-1) * (2*bH-1) * (2*bW-1) 行 × n_heads 列

例如 bT=12, bH=1, bW=1:
    表大小: (23) × (1) × (1) = 23 行 × 4 heads

索引: 长方体内部每个 token 对之间的相对坐标差值, 转为非负索引
```

### 6.6 前馈网络 (PositionwiseFFN)

```
x = x + ffn(norm(x))

ffn内部:
    Linear(C → 4*C)  [hidden=256 for dim=64, 512 for dim=128]
    → gelu 激活
    → Dropout(0.1)
    → Linear(4*C → C)
    → Dropout(0.1)
```

---

## 7. 解码器 (CuboidTransformerDecoder)

**代码位置**: `cuboid_transformer.py:2087-2440`

### 7.1 初始化

```python
z_init_method = "zeros"

Step 1: 创建零张量 (1, 26, 12, 12, 128)
Step 2: + 位置编码 (可学习 T+H+W embedding)
Step 3: Linear(128, 128) 投影
Step 4: expand → (B, 26, 12, 12, 128)
```

### 7.2 解碼流程 (从上到下, 2个 block)

```
Block 2 (i=1, 最深层, dim=128, T=12, H=12, W=12):
    dec_use_first_self_attn=False:
        → 跳过 self-attn, 直接跨注意力到 mem_l[1]

    CuboidCrossAttentionLayer:
        从 query (B, 26, 12, 12, 128) 跨注意 memory (B, 12, 12, 12, 128)
        配置: cross_1x1, n_temporal=1
        ┌──────────────────────────────────────────┐
        │ Q来自query (26步预测)                    │
        │ K,V来自memory (12步输入编码)             │
        │ cuboid_hw=(1,1): 逐像素跨注意力         │
        │ n_temporal=1: 26个Q时间步各自独立跨注意  │
        └──────────────────────────────────────────┘
    输出: (B, 26, 12, 12, 128)

↓ Upsample3DLayer
    目标: (26, 24, 24), 通道: 128→64
    最近邻插值 + Conv2D(3×3)
    → (B, 26, 24, 24, 64)

Block 1 (i=0, 最浅层, dim=64, T=12, H=24, W=24):
    depth=1:
        Layer 0:
            1. CuboidSelfAttentionLayer (axial, dim=64)
               → (B, 26, 24, 24, 64)
            2. CuboidCrossAttentionLayer (cross_1x1)
               → 跨注意 mem_l[0] (B, 12, 24, 24, 64)
               → (B, 26, 24, 24, 64)
    输出: (B, 26, 24, 24, 64)
```

### 7.3 跨注意力 (CuboidCrossAttentionLayer) 详解

**与 self-attention 的区别:**

| 特性 | Self-Attention | Cross-Attention |
|------|---------------|-----------------|
| Q来源 | 输入x | 输入x (query) |
| K,V来源 | 输入x | encoder memory |
| 时间维度 | query和memory必须相同T | query=26步, memory=12步 (通过n_temporal对齐) |
| 全局向量 | 可更新 | 只读取, 不更新 |

**n_temporal 机制**: 把时序分成 n_temporal 组，同一组内跨注意力。当前 `n_temporal=1` 表示所有时间步算一大组。

```python
# cross_1x1 模式 (最简)
cuboid_hw = (1, 1)     # 像素级注意力, 每个空间位置独立
shift_hw = (0, 0)       # 无窗口移位
strategy = ('l', 'l', 'l')  # 全局部策略
n_temporal = 1          # 时间不分組
```

---

## 8. 最终上采样解码 (FinalDecoder)

**代码位置**: `cuboid_transformer.py:2522-2580`

```
输入: (B, 26, 24, 24, 64)

Step 1: Upsample3DLayer
    目标尺寸: (26, 24, 48)  ← 恢复到原始输入分辨率
    注意: 经度从24恢复到48

    reshape(B*26, 24, 24, 64) → permute(B*26, 64, 24, 24)
    → Upsample(nearest): (24,24)→(24,48)
    → Conv2D(3×3, 64→64)
    → permute → reshape → (B, 26, 24, 48, 64)

Step 2: Conv2D 精细化 ×1 (GroupNorm + LeakyReLU)
    reshape → Conv2D(3×3, 64→64) → reshape
    输出: (B, 26, 24, 48, 64)

Step 3: 最终投影
    Linear(64, 4)
    输出: (B, 26, 24, 48, 4)   ← 4通道 (sst, t300, ua, va)
```

---

## 9. 损失函数与评估指标

### 9.1 训练损失

```python
loss = F.mse_loss(pred_seq, target_seq)
```

对**全部 4 通道、26 个时间步、全空间格点**做 MSE。无通道加权、无掩码。

### 9.2 SST MSE/MAE

```python
sst_pred = pred_seq[..., 0:1].contiguous()    # 只取SST通道
sst_target = target_seq[..., 0:1].contiguous()
self.valid_mse(sst_pred, sst_target)
self.valid_mae(sst_pred, sst_target)
```

使用 `torchmetrics.MeanSquaredError` / `MeanAbsoluteError` 累积全验证集计算。

### 9.3 Niño 3.4 指数计算

```python
def sst_to_nino(sst):
    # 空间平均 (Niño 3.4 区域: 5°S-5°N, 170°W-120°W)
    nino = sst[:, :, 10:13, 19:30].mean(dim=[2, 3])  # (B, 26)
    
    # 3个月滑动平均 (WMO标准)
    nino = nino.unfold(dim=1, size=3, step=1).mean(dim=2)  # (B, 24)
    # 26 - 3 + 1 = 24 个有效Niño值
    return nino
```

### 9.4 皮尔逊相关系数

```python
# 每个 lead month 独立计算
pred = nino_preds - nino_preds.mean(dim=0)
true = nino_true  - nino_true.mean(dim=0)
cor[i] = Σ(pred_i * true_i) / sqrt(Σ(pred_i²) * Σ(true_i²))

# 最终报告的值
test_corr_nino3.4 = mean(cor[0..23])   # 24个lead month平均
```

**24个lead month的解释:** 26个月SST → 3月滑动 → 24个Niño值。cor[0]是第1-3月SST平均的预测, cor[23]是第24-26月SST平均的预测。

### 9.5 加权相关系数

```python
weight[i] = [1.5]*4 + [2]*7 + [3]*7 + [4]*6 × log(i+1)
# 远期lead month权重更大 (更重视长期预测能力)
test_corr_weighted = mean(weight[i] * cor[i])
```

### 9.6 Niño RMSE

```python
rmse[i] = sqrt(mean((nino_pred[:,i] - nino_true[:,i])²))
test_nino_rmse = mean(rmse[0..23])
```

---

## 10. 优化器与学习率调度

### 10.1 AdamW

```
lr=1e-4, weight_decay=1e-5

参数分组:
    Group 1 (有衰减): 所有参数, 除LayerNorm权重和bias
        weight_decay = 1e-5
    Group 2 (无衰减): LayerNorm参数 + bias项
        weight_decay = 0
```

### 10.2 学习率调度

```
两阶段 SequentialLR:

Phase 1 — Warmup (前20%步数):
    lr = 0 → 1e-4   (线性增长)

Phase 2 — Cosine Annealing (后80%步数):
    lr = 1e-4 → 1e-7  (余弦退火)

总步数 = max_epochs × num_train_samples / total_batch_size
       = 100 × 56470 / 64 ≈ 88,230 steps
```

### 10.3 梯度裁剪

```yaml
gradient_clip_val: 1.0    # 最大范数1.0
```

---

## 11. 完整数据形状流转表

| 阶段 | 形状 | 说明 |
|------|------|------|
| 原始NC | (4645, 36, 24, 72) | CMIP原始文件 |
| 预处理后缓存 | (56508, 24, 48, 4) | CMIP6+CMIP5合并 |
| 每条样本 | (38, 24, 48, 4) | 12输入+26目标 |
| 模型输入 (in_seq) | (B, 12, 24, 48, 4) | 批次化后 |
| InitialEncoder后 | (B, 12, 24, 24, 64) | 经度减半, 通通扩 |
| Enc Block1后 | (B, 12, 24, 24, 64) | mem_l[0] |
| PatchMerge后 | (B, 12, 12, 12, 128) | H/W减半, 通道翻倍 |
| Enc Block2后 | (B, 12, 12, 12, 128) | mem_l[1] |
| Decoder初始化 | (B, 26, 12, 12, 128) | 零初始+位置编码 |
| Dec Block2后 | (B, 26, 12, 12, 128) | cross-attn到mem_l[1] |
| Upsample后 | (B, 26, 24, 24, 64) | 最近邻上采样 |
| Dec Block1后 | (B, 26, 24, 24, 64) | self+cross到mem_l[0] |
| FinalDecoder后 | (B, 26, 24, 48, 64) | 恢复到原始分辨率 |
| 最终投影 | (B, 26, 24, 48, 4) | 4通道输出 |
| Niño提取 | (B, 24) | SST→Niño 3.4区域→3月平滑 |

---

## 12. 可用的注意力策略清单

### 12.1 Self-Attention (编码器)

| 策略名 | 时间复杂度 | 空间复杂度 | 说明 |
|--------|-----------|-----------|------|
| `axial` | 低 | 低 | 时间/高度/宽度分轴注意力 (3个独立attention) |
| `divided_st` | 中 | 中 | 时间注意力 + 全空间注意力 (2个attention) |
| `full` | 极高 | 高 | 单长方体覆盖全 (T,H,W) — 仅小网格可用 |
| `video_swin_2x4` | 中 | 中 | 滑动窗口 (P=2, M=4), 第二次shift一半 |
| `spatial_lg_4` | 中高 | 中 | 局部(M,M) + 膨胀(M,M) + 时间轴, 适合空间纹理 |
| `axial_space_dilate_2` | 中 | 中 | 膨胀卷积式空间注意力 (K=2), 感受野更大 |

完整变体:
- `video_swin_{P}x{M}`: P∈{1,2,4,8,10}, M∈{1,2,4,8,16,32} → 38种
- `spatial_lg_{M}`: M∈{1,2,4,8,16,32} → 6种
- `axial_space_dilate_{k}`: k∈{2,4,8} → 3种

### 12.2 Cross-Attention (解码器)

| 策略名 | 说明 |
|--------|------|
| `cross_1x1` | 逐像素跨注意力, K=1x1局部 |
| `cross_4x4` | 4×4局部窗口跨注意力 |
| `cross_4x4_lg` | 4×4局部 + 4×4膨胀 |
| `cross_4x4_heter` | 4×4局部 + 膨胀 + 移位 (3层) |

### 12.3 配置方式

在 `cfg.yaml` 中:
```yaml
# 字符串 → 所有层相同
self_pattern: "axial"

# 列表 → 每层不同 (长度必须等于 enc_depth 的层数)
self_pattern: ["axial", "divided_st", "spatial_lg_4"]  
```

---

## 附录: 关键文件速查

| 文件 | 内容 |
|------|------|
| `src/earthformer/cuboid_transformer/cuboid_transformer.py` | 完整模型: 所有类定义 (3200+行) |
| `src/earthformer/cuboid_transformer/cuboid_transformer_patterns.py` | 注意力模式注册表 |
| `src/earthformer/cuboid_transformer/utils.py` | RMSNorm, padding, 初始化 |
| `src/earthformer/datasets/enso/enso_dataloader.py` | 数据加载与缓存 |
| `src/earthformer/metrics/enso.py` | Niño指标计算 |
| `scripts/cuboid_transformer/enso/train_cuboid_enso.py` | 训练入口 |
| `scripts/cuboid_transformer/enso/cfg.yaml` | 训练配置 |
| `scripts/datasets/preprocess_enso.py` | 数据预处理脚本 |
