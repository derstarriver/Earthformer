# 西北太平洋 SSTA 预测 — Earthformer 配置适配方案

> 输入: (14, 161, 241, 3) → 输出: (3, 161, 241, 1)

---

## 0. 现状与差距

| 项目 | ENSO (原) | NWP (目标) | 差距 |
|------|----------|-----------|------|
| 网格 | 24×48 = 1,152 | 161×241 = 38,801 | **34×** |
| 时间 | 12 月 | 14 天 | 尺度不同 |
| 通道入 | 4 (sst, t300, ua, va) | 3 (ssta, u10, v10) | — |
| 通道出 | 4 | 1 (仅 ssta) | — |
| 编码层 | 2 层 | 3 层 | 需加深 |
| 注意力 | axial (全轴) | spatial_lg (分块) | 需切换 |

---

## 1. Initial Downsample — 空间降维的第一步

### 1.1 源码分析

**位置**: `cuboid_transformer.py` 第 2442-2520 行, `InitialEncoder` 类

```
输入 (B, T, H, W, C_in)
  → reshape(B*T, H, W, C) → permute → (B*T, C, H, W)
  → K×Conv2D(3×3, GroupNorm+LeakyReLU)   # 保持空间尺寸
  → permute回 → reshape(B, T, H, W, C_out)
  → PatchMerging3D(downsample=scale)      # 空间降采样
```

`downsample_scale` 的处理：
- `int` → `(1, scale, scale)` — 仅空间
- `len=2` → `(1, h_scale, w_scale)`
- `len=3` → 直接使用

### 1.2 PatchMerging3D 机制

**位置**: 第 211-296 行

输入 `(B,T,H,W,C)`、下采样因子 `(dt,dh,dw)`：
1. 填充到整除
2. reshape → `(B, T//dt, dt, H//dh, dh, W//dw, dw, C)`
3. permute + merge → `(B, T//dt, H//dh, W//dw, dt·dh·dw·C)`
4. LayerNorm → Linear → `out_dim`

### 1.3 推荐配置

```yaml
initial_downsample_type: "conv"
initial_downsample_scale: [1, 4, 4]    # H 161→40, W 241→60
initial_downsample_conv_layers: 3      # 原2→3，更多初始特征提取
```

**计算**：
```
161 / 4 = 40.25 → pad到164 → 164/4 = 41
241 / 4 = 60.25 → pad到240 → 240/4 = 60
After InitialEncoder: (B, 14, 41, 60, 64)
```

`FinalDecoder` 的 `Upsample3DLayer` 使用 `nn.Upsample(size=target)` 而非 `scale_factor`，因此可以处理非整数倍率的恢复 (41→161)。

**注意**：`CuboidTransformerModel.__init__` 第 2912 行有断言 `H_in == H_out and W_in == W_out`。你的输入输出空间相同 (161×241 → 161×241)，不触发此限制，无需修改源码。

---

## 2. base_units — 控制模型容量与显存

### 2.1 源码分析

**位置**: `CuboidTransformerEncoder.__init__` 第 1758-1763 行

```python
block_units = [
    round_to(base_units * int((max(downsample) ** scale_alpha) ** i), 4)
    for i in range(num_blocks)
]
```

对于 `downsample=2, scale_alpha=1.0`：
- i=0: `64 * 1 = 64`
- i=1: `64 * 2 = 128`
- i=2: `64 * 4 = 256`

### 2.2 对比评估

| base_units | Enc Block 0 | Block 1 | Block 2 | 参数量 (est.) | 显存/样本 | 
|-----------|------------|---------|---------|-------------|----------|
| 32 | 32 | 64 | 128 | ~0.5M | 低 |
| 48 | 48 | 96 | 192 | ~1.0M | 中 |
| **64** | **64** | **128** | **256** | **~1.8M** | **中高** |

### 2.3 推荐

**`base_units: 64`**。理由：
- 1.8M 参数对 A6000 (48GB) 绰绰有余
- attention head_dim = 64/4 = 16，维度足够表达 SST 模态
- 原 ENSO 任务已在此配置下验证收敛良好

---

## 3. 编码器深度与注意力策略

### 3.1 推荐配置

```yaml
enc_depth: [2, 2, 2]   # 3层，每层2个attention block
dec_depth: [2, 2, 2]
downsample: 2

# 逐层注意力策略
self_pattern: ["axial", "spatial_lg_8", "divided_st"]
cross_self_pattern: ["axial", "spatial_lg_8", "divided_st"]
cross_pattern: ["cross_1x1", "cross_1x1", "cross_1x1"]
```

### 3.2 分辨率变化链

```
输入 (14, 161, 241, 3)
  → InitialEncoder (1,4,4):  (14, 41, 60, 64)
  
Block 0 (axial, dim=64):     (14, 41, 60, 64)   → mem[0]
  → PatchMerge (1,2,2):      (14, 20, 30, 128)

Block 1 (spatial_lg_8, dim=128): (14, 20, 30, 128) → mem[1]
  → PatchMerge (1,2,2):          (14, 10, 15, 256)

Block 2 (divided_st, dim=256):   (14, 10, 15, 256) → mem[2]

解码器从 (14, 10, 15, 256) 开始，逐层上采样回 (3, 161, 241, 64)
  → dec_final_proj: Linear(64, 1) → (3, 161, 241, 1)
```

### 3.3 为什么逐层切换注意力

| 层级 | 形状 | 策略 | 原因 |
|------|------|------|------|
| Block 0 | 41×60 | `axial` | 最大空间，轴向分解最省计算 |
| Block 1 | 20×30 | `spatial_lg_8` | 中等空间，8×8局部+膨胀兼顾纹理和感受野 |
| Block 2 | 10×15 | `divided_st` | 最小空间，时空分离聚焦动力学 |

### 3.4 axial vs spatial_lg 对显存的影响

对于 `(14, 41, 60, 64)`，`axial` 模式：
- 最大序列长度 = max(14, 41, 60) = 60
- 注意力矩阵 `(B, 4, 60, 60)` — 极小，完全可行

对于 `(14, 20, 30, 128)`，`spatial_lg_8` 模式：
- 切分成 8×8 方块，每块 QKV 64×64 — 可行

---

## 4. 解码器 — 上采样恢复

### 4.1 当前机制

**位置**: `CuboidTransformerDecoder` 第 2087-2440 行 + `FinalDecoder` 第 2522-2580 行

```
解码器块之间: Upsample3DLayer(nearest + Conv2D)
最终恢复:     FinalDecoder = Upsample3DLayer + K×Conv2D
最终投影:     dec_final_proj: Linear(64, 1)
```

### 4.2 推荐

```yaml
dec_use_first_self_attn: false   # 顶层先cross-attn再self-attn
dec_hierarchical_pos_embed: true # 每层上采样后重加位置编码
final_upsample_conv_layers: 2    # 原1→2，更平滑的上采样
```

**不需要修改源码**：`Upsample3DLayer` 用 `nn.Upsample(size=target)` 而非 `scale_factor`，自动处理非整数倍率 (41→161)。

---

## 5. Residual Prediction

### 5.1 评估

| 维度 | 预测 SSTA(绝对值) | 预测 ΔSSTA(增量) |
|------|------------------|------------------|
| 日尺度自相关 | r(t,t+1)≈0.95 | r(Δt, Δt+1)≈0.1 **← 好训练** |
| 数值范围 | ~[-3, 3] °C | ~[-1, 1] °C **← 更窄** |
| 物理意义 | 保留绝对温度信息 | 聚焦变化趋势 |
| 持久性baseline | SSTA(t+3)≈SSTA(t) (太强) | ΔSSTA≈0 (有意义) |

### 5.2 推荐

**当前先用 SSTA 绝对值**。理由：
1. 数据集已实现，改增量需重跑预处理
2. 14天输入窗口已足够短，持久性不算压倒性优势
3. 训练成熟后再快速实现增量版本对比

如需切换，仅需改 Dataset:

```python
# __getitem__ 中
y_residual = y - x[-1:, ..., 0:1]  # 最后一天的SSTA
return x, y_residual, mask
```

---

## 6. 最终推荐配置

```yaml
model:
  base_units: 64
  scale_alpha: 1.0
  enc_depth: [2, 2, 2]
  dec_depth: [2, 2, 2]
  enc_use_inter_ffn: true
  dec_use_inter_ffn: true
  dec_hierarchical_pos_embed: true
  downsample: 2
  downsample_type: "patch_merge"
  upsample_type: "upsample"

  # 全局向量 — 大网格下开启
  num_global_vectors: 8
  use_dec_self_global: true
  dec_self_update_global: true
  use_dec_cross_global: true
  use_global_vector_ffn: true
  use_global_self_attn: false
  separate_global_qkv: false
  global_dim_ratio: 1

  self_pattern: ["axial", "spatial_lg_8", "divided_st"]
  cross_self_pattern: ["axial", "spatial_lg_8", "divided_st"]
  cross_pattern: ["cross_1x1", "cross_1x1", "cross_1x1"]
  dec_cross_last_n_frames: null

  attn_drop: 0.1
  proj_drop: 0.1
  ffn_drop: 0.1
  num_heads: 4
  ffn_activation: "gelu"
  gated_ffn: false
  norm_layer: "layer_norm"
  padding_type: "zeros"
  pos_embed_type: "t+h+w"
  use_relative_pos: true
  self_attn_use_final_proj: true
  dec_use_first_self_attn: false
  z_init_method: "zeros"

  initial_downsample_type: "conv"
  initial_downsample_activation: "leaky"
  initial_downsample_scale: [1, 4, 4]
  initial_downsample_conv_layers: 3
  final_upsample_conv_layers: 2
  checkpoint_level: 0

  attn_linear_init_mode: "0"
  ffn_linear_init_mode: "0"
  conv_init_mode: "0"
  down_up_linear_init_mode: "0"
  norm_init_mode: "0"

optim:
  total_batch_size: 16            # 网格34×更大 → batch缩小
  micro_batch_size: 2
  method: "adamw"
  lr: 0.0001
  wd: 1.0e-05
  max_epochs: 50
  warmup_percentage: 0.1
  lr_scheduler_mode: "cosine"
  min_lr_ratio: 1.0e-3
  early_stop: true
  early_stop_patience: 10
  save_top_k: 3
```

---

## 7. 修改检查清单

| # | 文件 | 修改内容 | 必要性 |
|---|------|---------|--------|
| 1 | `cfg.yaml` | 全部 model/optim 参数 | **必须** |
| 2 | `nw_pacific_dataset.py` | `build_dataloaders()` 入口 | **必须** (已有) |
| 3 | 新训练脚本 | 基于 `train_cuboid_enso.py` 改造 | **必须** |
| 4 | `cuboid_transformer.py:2912` | `H_in==H_out` 断言 | **不需要** (你的数据输入输出同分辨率) |
| 5 | `cuboid_transformer.py:280` | `pad_t or pad_h or pad_w` bug | 可选 (不影响 spatial-only downsample) |
