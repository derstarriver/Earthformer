# Earthformer SSTA 预测 — 损失函数详解

> 对应源码: `scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py`

---

## 目录

1. [数据形状约定](#1-数据形状约定)
2. [海洋掩码机制](#2-海洋掩码机制)
3. [训练/验证 Loss 公式](#3-训练验证-loss-公式)
4. [逐步骤数值追踪](#4-逐步骤数值追踪)
5. [验证集 Metrics (torchmetrics)](#5-验证集-metrics-torchmetrics)
6. [测试集评估](#6-测试集评估)
7. [Loss / Metrics 关系对照表](#7-loss--metrics-关系对照表)
8. [已知问题与影响](#8-已知问题与影响)

---

## 1. 数据形状约定

| 符号 | 含义 | 值 |
|------|------|-----|
| `B` | micro batch size | 2 |
| `T_in` | 输入时序长度（天） | 14 |
| `T_out` / `T` | 输出时序长度（天） | 3 |
| `H` | 纬度网格 | 161 |
| `W` | 经度网格 | 241 |
| `C_in` | 输入通道 | 3 (ssta, u10, v10) |
| `C_out` | 输出通道 | 1 (ssta) |
| `N_ocean` | 单样本海洋格点数 | ≈ 34,879 (~90%) |
| `accum` | 梯度累积步数 | 8 (total_bs=16 / micro_bs=2) |

tensor 形状流转:

```
X 输入:      (B, 14, 161, 241, 3)
Y 标签:      (B,  3, 161, 241, 1)
pred 预测:   (B,  3, 161, 241, 1)
mask (原始): (161, 241, 1)           # 所有样本共享同一掩码
mask_t:      (B, 1, 161, 241, 1)     # 广播后
```

---

## 2. 海洋掩码机制

### 2.1 掩码来源

```python
# generate_ocean_mask.py → mask.npy
mask.shape = (161, 241, 1)    # dtype=float32
mask = 1.0  → 海洋
mask = 0.0  → 陆地
```

### 2.2 掩码在数据管线中的角色

```
1. 原始数据处理:
   ssta_raw, u10_raw, v10_raw  →  形状 (T, 161, 241)

2. 归一化 (基于海洋像素统计):
   ssta = (ssta_raw - mean_ocean) / std_ocean
   u10  = (u10_raw  - mean_ocean) / std_ocean
   v10  = (v10_raw  - mean_ocean) / std_ocean

3. 陆地归零 (归一化之后):
   ssta[mask == 0] = 0.0
   u10[mask == 0]  = 0.0
   v10[mask == 0]  = 0.0
```

> **关键设计**: 先归一化、后掩码。若先掩码（陆地→0），再归一化，陆地风速变成 `(0-mean)/std ≠ 0`，陆地会在数据中产生非零信号。当前顺序保证陆地 ≡ 0。

### 2.3 掩码在 Loss 中的广播

```python
mask_t = mask.reshape(B, 1, H, W, 1)
# mask_t: (B, 1, 161, 241, 1)
# pred:   (B, 3, 161, 241, 1)
# 相乘时自动广播 (B,1,161,241,1) → (B,3,161,241,1)
```

---

## 3. 训练/验证 Loss 公式

### 3.1 公式

```python
mask_t = mask.reshape(B, 1, mask.shape[1], mask.shape[2], 1)

loss = ((pred - Y) ** 2 * mask_t).sum() / (mask.sum() * T)
```

### 3.2 拆解

```
Step 1: (pred - Y)²                    → (B, 3, 161, 241, 1)    所有像素的平方误差
Step 2: * mask_t                        → (B, 3, 161, 241, 1)    陆地误差 → 0
Step 3: .sum()                          → 标量                    全部 B×3×161×241 求和
Step 4: / mask.sum()                    → 标量                    除以单样本海洋格点数
Step 5: / T                             → 标量                    除以预测天数
```

展开为数学表达式：

$$\text{loss} = \frac{\sum_{b=1}^{B} \sum_{t=1}^{T} \sum_{h=1}^{H} \sum_{w=1}^{W} (pred_{b,t,h,w} - Y_{b,t,h,w})^2 \cdot mask_{h,w}}{N_{ocean} \cdot T}$$

其中 $N_{ocean} = \sum_{h,w} mask_{h,w} \approx 34,879$

### 3.3 数值示例

假设一个 batch (B=2, T=3):

| 步骤 | 计算 | 结果 |
|------|------|------|
| 分子 .sum() | 2 × 3 × 34,879 × (平均每像素误差) | ≈ 209,274 × mean_err² |
| 分母 mask.sum() | 1 × 34,879 | 34,879 |
| 分母 T | 3 | — |
| loss | 209,274 × mean_err² / (34,879 × 3) | **2 × mean_err²** |

> 由于分子含 B 个样本但分母没有除以 B，loss = B × (单样本平均MSE) = 2 × mean_err²

---

## 4. 逐步骤数值追踪

以一个具体 batch 为例，假设某个验证步的输出值：

```
pred (B=2, 3, 161, 241, 1):
  - ocean pixels:  预测值（如 ~0.1 normalized SSTA）
  - land pixels:   任意值（如 ~0.05，模型无约束）

Y (B=2, 3, 161, 241, 1):
  - ocean pixels:  真实 SSTA 归一化值（如 ~0.12）
  - land pixels:   0.0（掩码后已归零）
```

计算过程：

```
Step 1 — 平方误差:
  (pred - Y)²: (2, 3, 161, 241, 1)
  海洋区: (0.1 - 0.12)² = 0.0004
  陆地区: (0.05 - 0.0)² = 0.0025  ← 非零！陆地预测有误差

Step 2 — 掩码:
  × mask_t → 陆地误差全部归零
  海洋误差保持不变

Step 3 — 求和:
  .sum() = 2 × 3 × 34,879 × 0.0004 = 83.71
  （仅海洋部分被计入）

Step 4 — 除以 N_ocean:
  83.71 / 34,879 = 0.0024

Step 5 — 除以 T:
  0.0024 / 3 = 0.0008

最终 loss = 0.0008 ← 实际是 B × 真实单样本MSE = 2 × 0.0004 = 0.0008
```

---

## 5. 验证集 Metrics (torchmetrics)

### 5.1 代码

```python
def validation_step(self, batch, batch_idx):
    ...
    pred_ocean = pred * mask_t      # (B, 3, 161, 241, 1)
    Y_ocean   = Y * mask_t          # 陆地强制为 0
    self.valid_mse(pred_ocean, Y_ocean)
    self.valid_mae(pred_ocean, Y_ocean)

def validation_epoch_end(self, outputs):
    self.log('valid_mse_epoch', self.valid_mse.compute())
    self.log('valid_mae_epoch', self.valid_mae.compute())
    self.valid_mse.reset()
    self.valid_mae.reset()
```

### 5.2 torchmetrics.MeanSquaredError 公式

$$\text{valid\_mse} = \frac{1}{B \cdot T \cdot H \cdot W \cdot 1} \sum (\text{pred\_ocean} - \text{Y\_ocean})^2$$

**关键**: torchmetrics 对所有元素求均值，包括陆地（陆地误差=0）。

### 5.3 loss vs valid_mse_epoch 的关系

| 指标 | 分子 | 分母 | 值 |
|------|------|------|-----|
| **loss** | $\sum_{B,T,H,W}$ | $N_{ocean} \cdot T$ | $B \cdot \overline{MSE}_{ocean}$ |
| **valid_mse** | $\sum_{B,T,H,W}$ | $B \cdot T \cdot H \cdot W$ | $\overline{MSE}_{all}$ (含陆地0) |

两者关系：

$$\text{valid\_mse\_epoch} = \text{loss} \cdot \frac{N_{ocean}}{B \cdot H \cdot W}$$

代入数值：
$$\text{valid\_mse\_epoch} \approx \text{loss} \cdot \frac{34,879}{2 \cdot 161 \cdot 241} \approx \text{loss} \cdot 0.45$$

示例：若 `valid_loss = 0.20`，则 `valid_mse_epoch ≈ 0.090`

> **验证据此选择 checkpoint**:
> ```python
> ModelCheckpoint(monitor="valid_mse_epoch", mode="min")
> EarlyStopping(monitor="valid_mse_epoch", patience=10, mode="min")
> ```

---

## 6. 测试集评估

### 6.1 分天聚合

```python
def test_step(self, batch, batch_idx):
    ...
    sq_err = ((pred - Y) ** 2 * mask_t).sum(dim=(0, 2, 3, 4))   # (3,) 每天独立
    abs_err = ((pred - Y).abs() * mask_t).sum(dim=(0, 2, 3, 4))  # (3,)
    n_ocean = mask_t.sum()                                        # 标量，单样本海洋格点数
    return {'sq_err': sq_err, 'abs_err': abs_err, 'n_ocean': n_ocean}
```

**形状追踪**: `.sum(dim=(0,2,3,4))`

```
输入: (B=2, T=3, H=161, W=241, C=1)
      沿 dim 0(B) 2(H) 3(W) 4(C) 求和
输出: (3,)  — 第0维(T)保留，每天一个标量
```

### 6.2 汇总与转换摄氏度

```python
def test_epoch_end(self, outputs):
    sq_err = torch.stack([o['sq_err'] for o in outputs]).sum(dim=0)   # (3,)
    abs_err = torch.stack([o['abs_err'] for o in outputs]).sum(dim=0) # (3,)
    n_total = sum(o['n_ocean'] for o in outputs)  # 所有样本海洋格点总数

    mse_per_day = sq_err / n_total      # (3,) 归一化 MSE
    mae_per_day = abs_err / n_total     # (3,) 归一化 MAE

    # 转换摄氏度
    ssta_std = float(stats['ssta_std'])   # 约 0.85°C
    mse_degC = mse_per_day * (ssta_std ** 2)
    rmse_degC = torch.sqrt(mse_degC)
    mae_degC = mae_per_day * ssta_std
```

### 6.3 测试 vs 训练的归一化差异

| | 训练/验证 loss | 测试 metrics |
|------|------|------|
| 归一化方式 | `/ (N_ocean * T)` | `/ (total_ocean_pixels)` |
| 包含 B | **否** (缺除以B) | **是** (n_total 累加了所有样本) |
| 分天 | 否 (整体除以T) | 是 (dim=0保留) |
| 输出单位 | 归一化 MSE | 归一化 MSE + °C |

**测试 metrics 是正确的**，因为它累加了所有样本的海洋格点数：

```
n_total = Σ(每个batch: mask_t.sum())
        = Σ(B × N_ocean)
        = num_batches × B × N_ocean
        = total_samples × N_ocean
```

---

## 7. Loss / Metrics 关系对照表

以一次验证为例（B=2, val_dataset=349 样本，约 175 个 batch）：

| 名称 | 记录位置 | 含义 | 正确性 |
|------|---------|------|--------|
| `train_loss` | metrics.csv | `sum / (N_ocean * T)` | ⚠️ 缺 ÷B |
| `valid_loss` | metrics.csv | 同公式，验证集均值 | ⚠️ 缺 ÷B |
| `valid_mse_epoch` | metrics.csv | torchmetrics mean 全像素 | ✅ 正确 |
| `valid_mae_epoch` | metrics.csv | torchmetrics mean 全像素 | ✅ 正确 |
| `test_mse_epoch` | test_metrics.csv | `sq_err / n_total` 纯海洋 | ✅ 正确 |
| `test_mae_epoch` | test_metrics.csv | `abs_err / n_total` 纯海洋 | ✅ 正确 |

### 数值换算关系

```
valid_loss (csv)         ≈  B × 单样本海洋MSE
valid_mse_epoch (csv)    ≈  valid_loss × (N_ocean / (B × H × W))
                         ≈  valid_loss × 0.45
test_mse_norm            =  纯海洋MSE，与 valid_loss/B 可直接比较
test_rmse_°C             =  sqrt(test_mse_norm) × ssta_std
```

### 换算示例

假设某 epoch 的 csv 记录为 `valid_loss = 0.2000`:

```
单样本海洋MSE ≈ 0.2000 / 2          = 0.1000  (归一化)
valid_mse_epoch ≈ 0.2000 × 0.45     = 0.0900  (含陆地0的均值)
预期 test RMSE  ≈ sqrt(0.1000) × 0.85 = 0.27°C
```



## 8. 已知问题与影响

### 问题: 训练/验证 loss 缺少除以 B

**代码 (line 412)**:
```python
loss = ((pred - Y) ** 2 * mask_t).sum() / (mask.sum() * T)
#                                          ^^^^^^^^^^^^^^^^
#                                          分母没有 B
```

**修复方案**:
```python
loss = ((pred - Y) ** 2 * mask_t).sum() / (mask.sum() * B * T)
```

### 影响评估

| 方面 | 严重程度 | 说明 |
|------|---------|------|
| loss 绝对值 | 低 | 值是正确值的 B=2 倍，但不影响 train/val 比较 |
| 梯度缩放 | **中** | 等效 lr = 设定值 × B = 2×，累积后 ≈ 16× per-sample |
| 与论文对比 | 低 | 只要明确除以B即可统一口径 |
| 过拟合贡献 | 低 | 训练/验证同公式，gap 相对关系不受影响 |
| checkpoint选择 | 无 | 使用 `valid_mse_epoch` (torchmetrics, 正确实现) |

### 等效学习率计算

```
设定 lr             = 1e-4
loss 缺 ÷B          → 梯度 ×2   → 等效 2e-4
accumulate_grad=8   → 梯度 ×8   → 等效 1.6e-3 (总)
```

> 实际上 PyTorch Lightning 的 `accumulate_grad_batches` 在 `backward()` 之后做了 loss 平均，所以严格来讲等效 lr 就是设定值 × B ≈ 2e-4。

### 海洋像素比例验证

```
H × W = 161 × 241 = 38,801
N_ocean ≈ 34,879 (约 90%)
N_land  ≈  3,922 (约 10%)
```

`mask.sum()` 可通过以下方式验证:
```python
import numpy as np
mask = np.load("datasets/SST-PREDICT/mask.npy")
print(f"ocean={mask.sum():.0f}, land={(1-mask).sum():.0f}")
```
