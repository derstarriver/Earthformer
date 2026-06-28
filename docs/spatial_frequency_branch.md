# Spatial Frequency Branch — 设计文档

## 动机

西北太平洋 SSTA 预测涉及多种周期性海洋现象（中尺度涡旋、Rossby 波、Kelvin 波、季节循环、ENSO 遥相关），这些现象在频域具有清晰的物理签名。Earthformer 的 cuboid transformer 通过注意力机制隐式学习这些模式，但缺乏显式的频域结构表示。

本模块在 InitialEncoder 之后插入一个轻量频率分支，用 **~6K 参数（<0.06%）** 提供显式的、物理可解释的频域增强。

## 插入位置

```
Input (B, 14, 161, 241, 4)
  ↓
InitialEncoder (Conv + PatchMerge)
  ↓ → (B, 14, 41, 61, 64)
  ↓
★★★ SpatialFrequencyBranch ★★★
  ↓ → (B, 14, 41, 61, 64)  形状不变
  ↓
enc_pos_embed
  ↓
Encoder → Decoder → ...
```

选择在 InitialEncoder 之后、enc_pos_embed 之前插入的理由：
- 空间已降至 41×61，FFT 计算量合理
- 在位置嵌入之前，让频率增强后的特征再获得空间位置信息
- 在编码器之前，transformer 注意力可以利用增强后的特征

## 架构

```
x: (B, T, H, W, D=64)
  │
  ├─────────────────────────────────────────────────┐              
  │                                                 │              
  ▼                                                 │              
  │ rFFT2(H, W)                                     │              
  │    X_f: (B, T, H, W_f, D) complex               │              
  │    amp = |X_f|, phase = ∠X_f                    │              
  │                                                 │              
  │ ┌─ State Conditioner ─────────────────────────┐ │              
  │ │  x → chunk(4, dim=-1) ⊲ 4 groups of 16 chs  │ │              
  │ │    → per-group GlobalAvgPool(T,H,W)          │ │              
  │ │    → concat → MLP(64→16→K)                  │ │              
  │ │    → softmax → band_w: (B, K)               │ │              
  │ └──────────────────────────────────────────────┘ │              
  │                     │                            │              
  │                     ▼                            │              
  │ ┌─ Spectral Gating ───────────────────────────┐ │              
  │ │  profile[k,h,w]  ∈ R^K×H×W_f (learnable)   │ │              
  │ │  gate[k,d]       ∈ R^K×D     (learnable)    │ │              
  │ │                                              │ │              
  │ │  logits[b,h,w,d,k] = band_w[b,k]            │ │              
  │ │                    × profile[k,h,w]          │ │              
  │ │                    × σ(gate[k,d])            │ │              
  │ │                                              │ │              
  │ │  attn = softmax_k(logits)    ⊲ 频段间归一化   │ │              
  │ │  weight[b,h,w,d] = Σ_k attn × profile[k,h,w] │ │              
  │ └──────────────────────────────────────────────┘ │              
  │                     │                            │              
  │                     ▼                            │              
  │ amp'  = amp ⊙ weight          ⊲ 频域直接 gating  │              
  │ X'    = amp' × exp(i·phase)   ⊲ 原相位重建       │              
  │ freq  = irfft2(X')            ⊲ 回到空域         │              
  │                                                 │              
  │                     ▼                            │              
  └───── output = x + α × freq_out ─────────────────┘              
  α: learnable scalar, init=0 → 训练初期分支无影响
```

## 核心设计决策

### 1. 仅 2D 空间 FFT（不做时间维度）

输入窗口仅 14 天，时间 FFT 频率分辨率极低（7 个频率分量）。空间域有 41×61 ≈ 2500 个频率分量，才是信息丰富的维度。未来若输入扩展至 28 天可重新评估 3D FFT。

### 2. 仅处理振幅，保留原始相位

相位在训练初期极易不稳定。保留 `torch.angle` 原值不动，仅对 `|X_f|` 做 gating，训练稳定性远优于振幅+相位联合处理。

### 3. State-conditioned（非 radius-conditioned）

频率分配不应由网格坐标决定，而应由海洋状态决定。

```
radius-based（不推荐）:   所有样本共享同一个频率分配
state-conditioned（采用）: GlobalAvgPool → 编码流域级状态 → 每样本动态分配
```

| 海洋状态 | band 激活模式 |
|---------|-------------|
| El Niño 成熟期 | 低频主导（流域级 ENSO 信号） |
| 涡旋活跃区 | 中频主导（中尺度涡旋） |
| 台风过境 | 高频增强（小尺度强混合） |

### 4. Channel-group split（物理分组）

将 64 通道等分为 4 组（16×4），分别池化后拼接入 MLP：

```
group_0 (SSTA-like)  → z0
group_1 (U10-like)   → z1
group_2 (V10-like)   → z2
group_3 (SLA-like)   → z3
        ↓
  concat → MLP → band_w
```

即使 InitialEncoder 已做跨通道混合，分组后的子空间仍保留对不同物理变量的敏感性，使 state conditioning 更具物理意义。

### 5. Softmax 频段间归一化（非无界加法）

```
weight = 1 + Σ(...)  → 无界，振幅可能爆炸或坍缩
softmax_k(logits)    → 有界，天然频率守恒，训练稳定
```

### 6. Entropy 正则（防 single-band collapse）

```python
loss_entropy = 1e-4 × (log(K) + Σ_k band_w[k] × log(band_w[k] + ε))
```

权重极小（1e-4），仅在模型试图 collapse 到单一频段且无充分数据支撑时才起作用。不影响主 loss 的收敛路径。

### 7. α 初始化为 0

```
训练 epoch 0：output = x + 0 × freq_out = x  → 等价于无分支
训练结束：α 自动学到最优值
```

确保频率分支不干扰 Earthformer 的原始训练轨迹，且可以安全插入已有 checkpoint 做 fine-tune。

## 参数量

| 组件 | 参数 |
|------|------|
| Per-group Linear(16→8), 4 groups | 544 |
| Linear(32→K), K=4 | 132 |
| freq_profiles: K×H×W_f | 5,084 |
| channel_gates: K×D | 256 |
| α | 1 |
| **合计** | **~6,017** |

相对于 Earthformer 总参数 ~10M，增加 **0.06%**。

## 论文可解释性

### 可视化-1：可学习的频段 profiles

```python
# 训练结束后可视化 4 个频段的空间签名
for k in range(K):
    imshow(freq_profiles[k])   # (H, W_f) → 该频段偏好的频率区域
```

预期结果：4 个 profile 自然收敛到从低频到高频的渐近分布，但边界是数据驱动的，非人为指定。

### 可视化-2：State-conditioned band activation

```python
# 对一组样本提取 band_w 并着色
band_w[b, :]  # 4 维 softmax 向量
```

预期结果：
- ENSO 正位相样本 → band_0 权重最高
- 涡旋区样本 → band_1/2 权重最高
- 风暴样本 → band_3 权重最高

### 可视化-3：Channel gate 的物理语义

```python
# 打印 σ(gate[k, :]) 的 64 维向量
# 按通道分组后查看到底哪些通道被该频段增强/抑制
```

### 论文段落模板

> *The frequency branch adapts its spectral modulation to the large-scale ocean state via a lightweight state conditioner. The 64-dimensional feature map is split into four channel groups, each pooled globally to capture basin-averaged statistics. A two-layer MLP maps the concatenated context vector to K frequency-band activation weights, enabling sample-specific enhancement or suppression of spectral components. A learnable spatial profile and per-channel gate for each band are combined with the state-conditioned weights via a softmax-normalized spectral attention, ensuring bounded and numerically stable modulation. An entropy regularization term (weight 1e-4) prevents pathological single-band collapse.*

## 代码

### 模块定义

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialFrequencyBranch(nn.Module):
    """State-conditioned adaptive spatial frequency gating for Earthformer.

    Insert between InitialEncoder and enc_pos_embed.
    """

    def __init__(self, dim: int = 64, num_bands: int = 4,
                 num_groups: int = 4, state_hidden: int = 16):
        super().__init__()

        assert dim % num_groups == 0
        group_dim = dim // num_groups

        # State conditioner: per-group pool → shared context MLP
        self.group_pools = nn.ModuleList([
            nn.Linear(group_dim, state_hidden // num_groups)
            for _ in range(num_groups)
        ])
        self.state_mlp = nn.Linear(state_hidden, num_bands)

        # Learnable frequency profiles & channel gates
        self.freq_profiles = nn.Parameter(torch.zeros(num_bands, 1, 1))
        self.channel_gates = nn.Parameter(torch.full((num_bands, dim), -2.0))

        # Global mixing scalar
        self.alpha = nn.Parameter(torch.zeros(1))

        self.dim = dim
        self.num_bands = num_bands
        self.num_groups = num_groups
        self._profiles_expanded = None

    def _ensure_profile(self, H: int, W_f: int, device: torch.device):
        if (self._profiles_expanded is not None
                and self._profiles_expanded.shape[-2:] == (H, W_f)):
            return self._profiles_expanded.to(device=device)

        p = self.freq_profiles                                     # (K, 1, 1)
        # Bilinear interpolate to target freq grid
        p = F.interpolate(p.unsqueeze(0), size=(H, W_f),
                          mode='bilinear', align_corners=False)    # (1, K, H, W_f)
        self._profiles_expanded = p.squeeze(0)                     # (K, H, W_f)
        return self._profiles_expanded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, D = x.shape
        W_f = W // 2 + 1

        # ── 2D spatial rFFT ──
        X_f = torch.fft.rfft2(x.float(), dim=(-3, -2))            # (B,T,H,W_f,D) complex
        amp = torch.abs(X_f)                                       # (B,T,H,W_f,D)
        phase = torch.angle(X_f)

        # ── State conditioner ──
        groups = x.chunk(self.num_groups, dim=-1)                  # each (B,T,H,W,D//G)
        z_parts = []
        for g, pool in zip(groups, self.group_pools):
            z = g.mean(dim=(1, 2, 3))                              # (B, D//G)
            z_parts.append(pool(z))                                # (B, H_hidden//G)
        context = torch.cat(z_parts, dim=-1)                       # (B, H_hidden)
        band_w = F.softmax(self.state_mlp(context), dim=-1)        # (B, K)

        # ── Spectral gating ──
        profiles = self._ensure_profile(H, W_f, x.device)          # (K, H, W_f)
        gates = torch.sigmoid(self.channel_gates)                  # (K, D)

        # logits[b,k,h,w,d] → softmax over k
        logits = (band_w[:, :, None, None, None]                   # (B, K, 1, 1, 1)
                  * profiles[None, :, :, :, None]                  # (1, K, H, W_f, 1)
                  * gates[None, :, None, None, :])                 # (1, K, 1, 1, D)
        attn = F.softmax(logits, dim=1)                            # (B, K, H, W_f, D)
        weight = (attn * profiles[None, :, :, :, None]).sum(dim=1) # (B, H, W_f, D)

        amp_enhanced = amp * weight                                # (B, T, H, W_f, D)

        # ── Reconstruct ──
        X_enhanced = amp_enhanced * torch.exp(1j * phase)
        freq_out = torch.fft.irfft2(X_enhanced, s=(H, W), dim=(-3, -2))
        freq_out = freq_out.to(x.dtype)

        return x + self.alpha * freq_out

    def entropy_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Optional entropy regularization (call externally, weight ~1e-4)."""
        groups = x.chunk(self.num_groups, dim=-1)
        z_parts = []
        for g, pool in zip(groups, self.group_pools):
            z = g.mean(dim=(1, 2, 3))
            z_parts.append(pool(z))
        context = torch.cat(z_parts, dim=-1)
        band_w = F.softmax(self.state_mlp(context), dim=-1)
        log_bw = torch.log(band_w + 1e-8)
        entropy = -(band_w * log_bw).sum(dim=-1).mean()
        K = self.num_bands
        return torch.log(torch.tensor(K, dtype=entropy.dtype,
                                      device=entropy.device)) - entropy
```

### 插入到 CuboidTransformerModel

在 `__init__` 中添加（`initial_encoder` 之后）：

```python
self.initial_encoder = InitialEncoder(...)
self.freq_branch = SpatialFrequencyBranch(dim=base_units)   # ← 新增
self.enc_pos_embed = PosEmbed(...)
```

在 `forward` 中：

```python
x = self.initial_encoder(x)
x = self.freq_branch(x)                                     # ← 新增
x = self.enc_pos_embed(x)
mem_l = self.encoder(x)
```

### 在主训练 loss 中加 entropy 正则

```python
# 在 training_step 中：
mse_loss = ((pred - Y) ** 2 * mask_t).sum() / (mask.sum() * B * T)
entropy_reg = self.torch_nn_module.freq_branch.entropy_loss(x)
total_loss = mse_loss + 1e-4 * entropy_reg
```

## 消融实验建议

| 实验 | 配置 | 验证目标 |
|------|------|---------|
| Baseline | 无频率分支 | 基准 |
| +Freq Branch (full) | 本模块完整版 | 整体收益 |
| − State conditioner | radius-based band assignment | 验证 state-conditioned 的价值 |
| − Channel split | 单一 GlobalAvgPool | 验证分组池化的价值 |
| − Softmax norm | `weight = 1 + Σ(logits)` | 验证 softmax 归一化的稳定性 |
| 仅前 50 轮比较 | Loss 曲线 | 验证 `α=0` init 是否加速收敛 |

## 限制与未来工作

1. **时间维度**：当前仅做 2D 空间 FFT。若输入窗口扩展至 28 天，3D FFT 值得重新评估。
2. **频段数量 K**：当前固定为 4。K 可作为超参搜索，或使用 nonparametric Bayesian 方法自动确定。
3. **跨尺度相互作用**：当前各频段独立 gating，跨尺度能量级联（turbulence cascade）未被显式建模。可增加 band-interaction MLP。
4. **训练初期**：`α=0` 保证安全，但 entropy regularization 需在 warmup 阶段之后才生效，避免早期 band 分布不稳定。
