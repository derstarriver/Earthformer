# Forced Spectral Learning Branch -- 设计文档

## 动机

西北太平洋 SSTA 预测涉及多种具有清晰频域签名的海洋现象（中尺度涡旋、Rossby 波、Kelvin 波、季节循环、ENSO 遥相关）。Earthformer 的 cuboid transformer 通过注意力隐式学习这些模式，但 attention 固有的低频偏好（softmax 平滑效应）使其容易丢失中高频细节（涡旋边界、锋面梯度）。

本模块实现 **Forced Spectral Learning**：用 FFT 提取中高频残差，以固定权重注入 encoder 输入特征，改变 QKV 计算结果，从而结构性改写 attention pattern。模块不可被 bypass。

**参数量: ~28K (0.28%)**。

## 与 V4 (SpatialFrequencyBranch) 的关键区别

| 维度 | V4 (旧) | V5 (当前) |
|------|---------|----------|
| 注入方式 | `x + alpha * freq_out`, alpha=0 init | `x + 0.3 * freq_delta`, 固定 beta=0.3 |
| 旁路风险 | alpha=0 时等价无分支 | 不可旁路 |
| 频谱选择 | 全频段处理 | soft band-pass, 偏高频 |
| State conditioner | band_w only | band_w (选择) + influence (强度) 解耦 |
| 梯度路径 | 仅 residual add | aux_head 直接梯度 + encoder QKV 梯度 |
| 正则化 | entropy_reg (防collapse) | smoothness_loss (防孤峰) |
| 输入连接 | 不 detach | 不 detach, 全梯度 |

## 插入位置

```
Input (B, 14, 161, 241, 4)
  ↓
InitialEncoder (Conv + PatchMerge)
  ↓ -> x: (B, 14, H', W', D)
  │
  ├── ForcedSpectralBranch ──> freq_delta
  │       │
  │       ├──> aux_head ──> aux_pred (aux loss, lambda=0.2)
  │       └──> smoothness_loss        (lambda=0.01)
  │
  ▼
x = x + 0.3 * freq_delta    <-- 固定权重, 不可旁路
  ↓
enc_pos_embed
  ↓
Encoder (QKV 计算含频率信息) -> Decoder -> Prediction
  ↓
total_loss = main + 0.2*aux + 0.01*smoothness
```

## 架构

```
x: (B, T, H, W, D=64)                    无 detach, 全梯度
  │
  ▼
  │ rFFT2(H, W)
  │    X_f: (B, T, H, W_f, D) complex
  │    amp = |X_f|, phase = angle(X_f)
  │
  │ ┌─ Soft band-pass (radius -> freq_weight) ────────┐
  │ │  r = sqrt(h_f^2 + w_f^2)  [frequency radius]     │
  │ │  r_norm in [0, 1]                                 │
  │ │  freq_weight = sigmoid(MLP(r_norm))  in (0,1)    │
  │ │    init: DC ~0.2, Nyquist ~0.7                    │
  │ │    -> 高频初权重高, 低频可被数据拉回               │
  │ └──────────────────────────────────────────────────┘
  │                     │
  │                     ▼
  │ ┌─ State Conditioner (two decoupled heads) ────────┐
  │ │  x -> chunk(4, dim=-1)  4 groups of 16 chs       │
  │ │    -> per-group GlobalAvgPool(T,H,W)              │
  │ │    -> concat -> Linear(64->16) -> context         │
  │ │                                                   │
  │ │  band_w  = softmax(Linear_sel(context))  (B,K)   │
  │ │    -> "which band" to activate                    │
  │ │  influence = sigmoid(Linear_str(context)) (B,K)  │
  │ │    -> "how much" modulation                       │
  │ │  band_w = band_w * (0.5 + influence)              │
  │ │    -> range [0.25, 1.25], init ~0.5               │
  │ └──────────────────────────────────────────────────┘
  │                     │
  │                     ▼
  │ ┌─ Spectral Gating ───────────────────────────────┐
  │ │  profile[k,h,w] in R^{K x H x W_f} (learnable)  │
  │ │  gate[k,d]      in R^{K x D}     (learnable)    │
  │ │                                                   │
  │ │  logits[b,k,h,w,d] = band_w[b,k]                 │
  │ │                    x profile[k,h,w]               │
  │ │                    x sigmoid(gate[k,d])           │
  │ │  attn_b = softmax_k(logits)    频段间归一化       │
  │ │  spectral_w[b,h,w,d] = sum_k attn_b * profile    │
  │ │                                                   │
  │ │  weight = spectral_w * freq_weight[h,w]          │
  │ │    -> 同时考虑 band gating (选择什么频段)          │
  │ │       和 band-pass (高低频偏好)                   │
  │ └──────────────────────────────────────────────────┘
  │                     │
  │                     ▼
  │ amp'  = amp * weight             频域直接 gating
  │ X'    = amp' * exp(i*phase)      原相位重建
  │ freq_delta = irfft2(X')          回到空域 (B,T,H,W,D)
  │
  │ ┌─ Auxiliary Head ───────────────────────────────┐
  │ │  Conv3d(D->D/2, (T_in-T_out+1,1,1))  T压缩     │
  │ │  Conv3d(D/2->D/4, 3x3)                         │
  │ │  Conv3d(D/4->C_out, 1x1)                        │
  │ │  -> aux_pred: (B, Tout, H, W, C_out)            │
  │ └──────────────────────────────────────────────────┘
  │
  │ ┌─ Smoothness Loss ──────────────────────────────┐
  │ │  |freq_delta_fft| mean over T,D -> (H, W_f)    │
  │ │  h_diff = mean((amp[1:,:]-amp[:-1,:])^2)        │
  │ │  w_diff = mean((amp[:,1:]-amp[:,:-1])^2)        │
  │ │  loss = 0.01 * (h_diff + w_diff)                │
  │ │  -> 防频谱孤峰, 不强制均匀分布                     │
  │ └──────────────────────────────────────────────────┘
  │
  ▼
return freq_delta, aux_pred, smoothness_loss


外部 (CuboidTransformerModel.forward):
  x = x + 0.3 * freq_delta       <-- QKV 自然包含频率信息
  x = enc_pos_embed(x)
  mem_l = encoder(x)              <-- K = W_k(x + 0.3*freq_delta)
                                       = W_k(x) + 0.3*W_k(freq_delta)
                                       等效于 attention logit bias
```

## 核心设计决策

### 1. x = x + beta * freq_delta, beta=0.3 固定（不可旁路）

```python
# 旧 (V4, 可旁路):
output = x + alpha * freq_out    # alpha=0 -> FFT无作用

# 新 (V5, 不可旁路):
x = x + 0.3 * freq_delta        # 0.3 固定, encoder必须适应
```

**等效性论证**: K = W_k(x + 0.3*freq_delta) = W_k(x) + 0.3*W_k(freq_delta)。QK^T = Q(W_k x)^T + 0.3*Q(W_k freq_delta)^T。第二项即为 attention logit 空间中的加性 bias，来源于频率内容。这避免了修改 cuboid attention 内部代码的复杂性，同时实现了等效的 attention logit bias 注入。

**为什么 beta 不学习**: 防止 encoder 学成 W_k 抑制 freq_delta 项（即 K 中的 freq 分量被 W_k 拉向零），绕过 FFT。固定 beta 保证 freq_delta 始终有非零贡献。

### 2. Soft band-pass（替代 hard high-pass）

```python
# 可学习频率选择函数: 低频可被"拉回"
freq_weight[r] = sigmoid(MLP(r_norm))

# 初始化:
#   DC (r=0):  sigmoid(-1.4) ~ 0.2  -> 低频初权重低
#   Nyq (r=1): sigmoid(0.85)  ~ 0.7  -> 高频初权重高
# 但 MLP 训练后可调整: 中频涡旋信号可以被拉高
```

**与 hard high-pass 对比**: hard split 无法适应不同 SST 状态（ENSO 期低频增强、涡旋季中频增强）。soft MLP 允许数据驱动调整。

### 3. Influence 与 band_w 解耦

```python
band_w   = softmax(state_mlp(context))         # "选哪个频段"
influence = sigmoid(influence_mlp(context))     # "调多少强度"
band_w   = band_w * (0.5 + influence)           # [0.25, 1.25]
```

**为什么解耦**: 同一个频段在不同样本中可能需要不同强度。例如 El Nino 期 band_0（低频）被选中，但强度应比 La Nina 期更大。解耦后 band_w 的 softmax 负责选择, influence 独立负责强度。

### 4. Auxiliary prediction head（直接梯度路径）

FFT branch 有独立的 aux_head 直接预测下采样后的 SST，计算 aux_loss。这为 freq_delta 提供了一个不经过 encoder/decoder 的梯度路径，确保 FFT 必须学到 predictive signal。

### 5. Smoothness loss（替代 entropy / diversity loss）

```python
loss_smooth = mean((amp相邻行差)^2 + (amp相邻列差)^2) * 0.01
```

- 只惩罚频谱中的孤立尖峰，不强制均匀分布
- SST 低频天然占主导，不违背物理
- 权重 0.01 极小，仅在最极端的孤峰情况下生效

### 6. 无 detach，全梯度

```python
freq_delta = FFT(x)     # x有梯度, 非 x.detach()
```

Encoder 可以通过梯度反馈帮助 FFT 学习更有用的频率特征。两者协作而非竞争。

## 参数量

| 组件 | 参数 |
|------|------|
| freq_weight_fn: Linear(1->16->1) | 1*16+16+16*1+1 = 49 |
| group_pools: 4x Linear(16->4) | 4*(16*4+4) = 272 |
| state_mlp: Linear(16->4) | 68 |
| influence_mlp: Linear(16->4) | 68 |
| freq_profiles: K x 1 x 1 | 4 |
| channel_gates: K x D | 4*64 = 256 |
| aux_head: 3x Conv3d | ~20K |
| **合计** | **~28,000** |

相对于 Earthformer 总参数 ~10M，增加 **0.28%**。

## 损失函数

```python
total_loss = main_loss + 0.2 * aux_loss + 0.01 * smoothness_loss

main_loss:      pred vs target, masked MSE, per-pixel-per-day
aux_loss:       aux_pred vs target_downsampled, MSE
smoothness:     anti spectral spike
```

| 项 | 权重 | 作用 |
|----|------|------|
| main_loss | 1.0 | 主预测任务 |
| aux_loss | 0.2 | 强制 freq_delta 学到 predictive signal |
| smoothness | 0.01 | 防频谱孤峰，不强制均匀分布 |

## 论文可解释性

### 可视化-1: 可学习频率选择曲线

```python
r = linspace(0, 1, 100)
freq_w = sigmoid(freq_weight_fn(r))
plot(r, freq_w)  # 训练前后对比
```

预期: 训练后中频 (0.3-0.6) 权重上升，对应涡旋尺度被增强。

### 可视化-2: State-conditioned band activation

```python
# ENSO 期样本 vs 正常期样本
band_w_enso.mean(dim=0) vs band_w_normal.mean(dim=0)
```

### 可视化-3: freq_delta 的 attention 影响

```python
# 对比有/无 freq_delta 时的 attention map
attn_base = softmax(Q @ K_base.T / sqrt(d))
attn_freq = softmax(Q @ K_freq.T / sqrt(d))
diff = (attn_freq - attn_base).abs().mean()
# 预期: 涡旋区 token 的 attention 分布被改变
```

### 论文段落模板

> *The Forced Spectral Learning Branch injects frequency-domain information directly into the encoder's QKV computation via feature-level modulation with a fixed mixing weight (beta=0.3). A soft band-pass mechanism, parameterized as a small MLP over normalized frequency radius, learns to emphasize mid-to-high spatial frequencies where attention-based feature smoothing loses detail. The band selection (softmax over K learnable frequency bands) and modulation strength (sigmoid-gated influence) are decoupled, allowing the model to independently choose which spectral band to activate and how strongly to modulate it per sample. An auxiliary prediction head provides a direct gradient path for the frequency residual, ensuring it carries predictive signal independent of the main encoder-decoder pathway. A spectral smoothness regularizer penalizes isolated frequency spikes without enforcing uniform spectral energy, respecting the natural low-frequency dominance in SST dynamics.*

## 消融实验建议

| 实验 | 配置 | 验证目标 |
|------|------|---------|
| Baseline | 无频率分支 | 基准 |
| +FSL (full) | 完整版 | 整体收益 |
| - aux_loss | lambda_aux = 0 | 验证直接梯度路径的必要性 |
| - soft band-pass | hard freq_weight = [0,1] | 验证可学习频率选择 |
| - influence | band_w = softmax only | 验证 influence 解耦的价值 |
| beta 敏感度 | beta in [0.1, 0.5, 0.7] | 验证注入强度 |

## 限制与未来工作

1. **时间维度**: 当前仅 2D 空间 FFT。输入扩展至 28 天时重新评估 3D FFT。
2. **K/V 分离调制**: 当前 QKV 共享特征增强。可对 K 和 V 分别用不同的 freq 投影。
3. **跨尺度耦合**: 当前各频段独立 gating。可建模 frequency cross-talk（如涡旋-平均流的能量交换）。
