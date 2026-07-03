# Earthformer 西北太平洋 SSTA 预测 — 工作流程详解

> 基于 `scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py` 和 `cfg_nwp.yaml`
> 最后更新: 7通道物理感知输入, 多物理损失, **自回归预测 + Scheduled Sampling + BPTT**

---

## 目录

1. [任务定义](#1-任务定义)
2. [数据管线](#2-数据管线)
3. [模型架构总览](#3-模型架构总览)
4. [初始卷积编码](#4-初始卷积编码)
5. [位置编码](#5-位置编码)
6. [编码器](#6-编码器)
7. [Cuboid Attention](#7-cuboid-attention)
8. [全局向量](#8-全局向量)
9. [解码器（单步）](#9-解码器单步)
10. [自回归展开与 Scheduled Sampling](#10-自回归展开与-scheduled-sampling)
11. [损失函数](#11-损失函数)
12. [评估指标](#12-评估指标)
13. [优化器与学习率调度](#13-优化器与学习率调度)
14. [数据形状流转表](#14-数据形状流转表)
15. [配置说明](#15-配置说明)
16. [训练流程](#16-训练流程)
17. [模型诊断套件](#17-模型诊断套件)
18. [已知问题与改进方向](#18-已知问题与改进方向)
19. [附录](#19-附录)

---

## 1. 任务定义

| 项目 | 值 |
|------|-----|
| 输入 | 14 天 x 161x241 x **7 通道** [ssta, u10, v10, sla, grad_x, grad_y, advection] |
| 输出 | 7 天 x 161x241 x 1 通道 [ssta] (自回归逐日展开) |
| 模型单步输出 | 1 天 x 161x241 x 1 通道 [delta SSTA] |
| 区域 | 10N-50N, 120E-180E |
| 分辨率 | 0.25 度 |
| 数据源 | ERA5 (SST/Wind) + CMEMS AVISO (SLA), 2001-2025 |
| 参数量 | ~10M (scale_alpha=0.4) |

---

## 2. 数据管线

### 2.1 预处理

| 步骤 | 脚本 | 输入/输出 |
|------|------|-----------|
| 空间裁剪 | `preprocess_nwp.py` | ERA5 raw -> SST_cropped.nc + Wind_cropped.nc |
| 海陆掩码 | `generate_ocean_mask.py` | -> mask.npy |
| 逐日气候态 | `compute_climatology.py` | -> climatology.nc |
| SSTA 计算 | `compute_ssta.py` | SST - climatology -> ssta.nc |
| SLA 预处理 | `preprocess_sla.py` | CMEMS AVISO -> SLA_cropped.nc |

### 2.2 训练时加载流程

```
1. 读取 ssta.nc + Wind_cropped.nc + SLA_cropped.nc + mask.npy
2. 通道独立 z-score 归一化 (训练集 2001-2022 统计)
3. 掩码陆地 -> 0
4. 计算物理派生通道 (仅海洋, 训练集统计数据归一化):
   grad_x    = d(SSTA)/dlon         东西向温度梯度
   grad_y    = d(SSTA)/dlat         南北向温度梯度
   advection = -(u10*dx + v10*dy)   风驱平流项
5. 堆叠 7 通道 -> (T, 161, 241, 7)
6. 按年切分:
   训练: 2001-2022 (stride=3, ~2672 样本)
   验证: 2023-2024 (stride=1, ~715 样本)
   测试: 2025       (stride=1, ~349 样本)
7. 滑窗采样: 14 天输入 + 7 天输出 = 21 天窗口
8. DataLoader -> (B,14,161,241,7), (B,7,161,241,1), mask
```

---

## 3. 模型架构总览

```
输入 (B, 14, 161, 241, 7)
   |
   v
[InitialEncoder]  Conv2Dx3 -> PatchMerge(1,4,4)
   |  (B, 14, 41, 61, 64)
   v
[PosEmbed]  t+h+w
   |
   v
[Encoder Block 0]  axial, dim=64          -> mem[0]  (B,14,41,61,64)
   |  PatchMerge(1,2,2)
   v
[Encoder Block 1]  spatial_lg_8, dim~80   -> mem[1]  (B,14,21,31,80)
   |  PatchMerge(1,2,2)
   v
[Encoder Block 2]  divided_st, dim~97     -> mem[2]  (B,14,11,16,97)
   |
   |  多尺度记忆: mem[0], mem[1], mem[2]
   |  全局向量: 8 x 64
   |
   v
[Decoder Init]  zeros(1, 11, 16, ~97) + PosEmbed
   |
   v
[Decoder Block 2]  cross->mem[2] -> self (divided_st)  -> upsample
   |
[Decoder Block 1]  cross->mem[1] -> self (spatial_lg_8) -> upsample
   |
[Decoder Block 0]  cross->mem[0] -> self (axial)
   |  (B, 1, 41, 61, 64)
   v
[FinalDecoder]  Upsample -> Conv2Dx2
   |  (B, 1, 161, 241, 64)
   v
[dec_final_proj]  Linear(64,1)
   |
   v
delta_SSTA  (B, 1, 161, 241, 1)  <- 单步预测

   在 _ar_forward 中循环 7 次:
   t=0: window_0 = X -> delta_0 -> pred_0 = X_last + delta_0
   t=1: window_1 = slide(window_0, next_ssta) -> delta_1 -> pred_1
   ...
   t=6: window_6 -> delta_6 -> pred_6

   -> preds = cat([pred_0..pred_6])  (B, 7, 161, 241, 1)
```

---

## 4. 初始卷积编码

```
输入: (B, 14, 161, 241, 7)
Step 1: reshape -> (B*14, 161, 241, 7) -> permute (B*14, 7, 161, 241)
Step 2: Conv2Dx3 (3x3, GroupNorm+LeakyReLU): 7->64, 64->64, 64->64
Step 3: reshape -> (B, 14, 161, 241, 64)
Step 4: PatchMerging3D(1,4,4): H:161->41, W:241->61
输出: (B, 14, 41, 61, 64)
```

---

## 5. 位置编码

pose_embed_type: "t+h+w"

三个独立的可学习 Embedding，分别沿时间、纬度、经度轴广播后相加。

---

## 6. 编码器

- 3 层层次结构, enc_depth=[2,2,2]
- scale_alpha=0.4: 通道 64->~80->~97
- 每层 2x CuboidSelfAttention + PositionwiseFFN
- 层间 PatchMerge(1,2,2) 2x 下采样

---

## 7. Cuboid Attention

| 层 | 模式 | cuboids | 目的 |
|---|------|---------|------|
| Block 0 | axial | T-full, H-full, W-full | 最大网格, 轴向分解省计算 |
| Block 1 | spatial_lg_8 | T-full, local 8x8, dilated 8x8 | 局部保纹理, 膨胀扩感受野 |
| Block 2 | divided_st | T-full, full HxW | 小网格全空间注意力 |
| Cross | cross_1x1 | 1x1 per-pixel | 像素级跨注意力, Q(解码)/K(记忆)/V(记忆) |

---

## 8. 全局向量

8 个可学习向量 (dim=64)，在 Encoder 中逐层传播更新，Decoder 中参与 self/cross attention，作为跨 cuboid 长距信息中转站。

---

## 9. 解码器（单步）

**关键改变**：与旧版并行解码不同，Decoder 只输出 1 帧:

```yaml
target_shape: [1, 161, 241, 1]   # 旧版是 [7, ...]
```

Decoder 初始化: `zeros(1, 11, 16, ~97) + PosEmbed`

3 个 Block 结构不变，但时间维 T_dec=1，cross-attention Q 来自单步 query，K/V 来自 14 步 encoder memory。最终投影 `Linear(64,1)` 输出单步 delta SSTA。

**为什么改为单步**: 旧版并行 7 帧中，Decoder 各 query 间无因果依赖，模型学到的最优解是"所有帧输出近似值"（ACF~0.99）。单步 Decoder 强制每一步的预测基于前一步的预测状态，建立了因果时序依赖。

---

## 10. 自回归展开与 Scheduled Sampling

### 10.1 推理流程 (Val/Test)

```python
window = X                              # (B, 14, H, W, 7)
preds  = []
for t in range(7):
    delta = model(window)               # 单步预测
    pred_t = window_last_ssta + delta   # 绝对 SSTA
    preds.append(pred_t)
    # 用自己的预测更新窗口
    next_ssta = pred_t.detach()         # 无梯度, 纯推理
    aux = window[-1, :, :, 1:4]         # u10/v10/sla 保持已知值
    gx, gy, adv = compute_physics(next_ssta, aux)
    next_frame = stack([next_ssta, aux, gx, gy, adv])
    window = cat([window[1:], next_frame])
return cat(preds)  # (B, 7, H, W, 1)
```

### 10.2 训练流程 (Scheduled Sampling + BPTT)

训练的关键区别在于 `next_ssta` 的来源:

```python
# 以概率 self._ss_prob 使用 teacher SSTA
if rand() < self._ss_prob:
    next_ssta = Y_teacher[:, t].detach()    # teacher forcing, 无跨步梯度
# 以概率 1-self._ss_prob 使用自己的预测
else:
    next_ssta = pred_t                      # 不 detach! -> BPTT 梯度路径
```

**Scheduled Sampling 衰减表**:

| epoch | P(teacher) | 效果 |
|-------|:----------:|------|
| 0 | 1.00 | 纯 teacher forcing, 稳定初始化 |
| 5 | 0.87 | 基本稳定, 偶尔自预测暴露 |
| 10 | 0.73 | 逐步增加自预测比例 |
| 15 | 0.60 | 半数 teacher, 开始建立 BPTT 路径 |
| 20 | 0.47 | 过半自预测, 模型适应自身误差 |
| 25 | 0.33 | 接近真实推理分布 |
| 30+ | 0.20 | 维持少量 teacher 防止崩溃 |

### 10.3 BPTT 梯度路径

当 scheduled sampling 选中"自预测"时:

```
step[t]  pred_t = ssta_last + model(window_t)        <- model params 参与
          |
          v (不 detach)
step[t+1] window_{t+1}.ssta = pred_t                 <- grad_fn 保留
          pred_{t+1} = ssta_last + model(window_{t+1})

d(loss)/d(params) +=
    d(loss)/d(pred_{t+1})
    * d(pred_{t+1})/d(window_{t+1})
    * d(window_{t+1})/d(pred_t)
    * d(pred_t)/d(params)                            <- 跨步梯度链
```

**与旧版的关键区别**: 旧版 AR v1 中, `window[:, 1:].detach()` 和 `aux.detach()` 截断了此路径。新版中, `window[:, 1:]` 来自 X 数据 (天然 leaf, 无 grad_fn, 不携带梯度图, 无需 detach), `aux` 同理。只有 `next_ssta` 在自预测模式时有 grad_fn, 通过 `torch.cat` 拼入 window, 建立跨步连接。

### 10.4 Gradient Checkpointing 与显存

训练时每个单步 forward 使用 `torch.utils.checkpoint`:

```python
delta = grad_ckpt(self.torch_nn_module, window, use_reentrant=False)
```

正向传播时丢弃中间激活, 反向传播时重新计算。跨步 BPTT 仅通过 ssta 通道 (B,H,W floats ~155KB per sample) 传递梯度, 模型内部激活通过 checkpoint 管理。总训练显存约为非 AR 模型的 1.3-1.5 倍 (13-16G)。

Val/Test 使用普通 `self.forward()` (无 checkpoint 开销), 显存与单步 forward 相同 (~9G)。

---

## 11. 损失函数

四项损失, 作用于 7 步完整 AR 轨迹:

| 损失项 | 公式 | 权重 | 目标 |
|--------|------|:---:|------|
| loss_mse | MSE(pred, Y) x day_weight[0.8->1.5] | 1.0 | 基础预测精度 |
| loss_grad | MSE(grad(delta), grad(delta_ref)) | 0.5 | 锋面/涡旋空间结构 |
| loss_tend | MSE(pred[t]-pred[t-1], Y[t]-Y[t-1]) | **5.0** | 逐日时间动力学 |
| loss_anti | relu(mse_plain - 0.95*persist_mse) | 0.3 | 反持续性退化 |

**loss_tend 为何在 AR 架构中有效**: 在并行解码中, pred[t] 和 pred[t+1] 由独立的 decoder query 生成, 无函数依赖, loss_tend 的梯度分别流向两个独立的 query embedding, 不触及 encoder memory 或 attention 的动力学编码。在 AR 架构 + BPTT 中, pred[t+1] 的计算图包含 pred[t] 作为输入, loss_tend 的梯度通过跨步链回传到模型参数, 真正逼迫模型学习"从状态 t 演化到 t+1"的机制。

---

## 12. 评估指标

### 训练/验证 (归一化空间)
- torchmetrics.MeanSquaredError / MeanAbsoluteError
- valid_loss = plain MSE (无物理分量), 用于 checkpoint 选择
- 额外 log: train_mse, train_grad, train_tend, train_anti, ss_prob

### 测试 (分天, 转换摄氏度)
- 每批累计 sq_err (T,) 和 abs_err (T,) over mask
- 全局 mse_per_day = sum(sq_err) / total_ocean_pixels
- 转 Celsius: rmse = sqrt(mse * ssta_std^2), mae = mae * ssta_std
- 保存 test_metrics.csv

---

## 13. 优化器与学习率调度

- AdamW: lr=1e-4, wd=1e-4
- 参数分组: LayerNorm/bias 无 wd, 其余有 wd
- 两阶段 LR: warmup 10% (0->1e-4) + cosine 90% (1e-4->1e-7)
- 正则: attn/proj/ffn dropout 0.2/0.2/0.3, grad_clip=1.0
- Early stop: patience=30 on valid_mse_epoch

---

## 14. 数据形状流转表

> scale_alpha=0.4, initial_downsample_scale=[1,4,4]

| 阶段 | 形状 | 说明 |
|------|------|------|
| 原始合并+归一化 | (9131, 161, 241, 7) | 7通道 |
| 模型输入 X | (B, 14, 161, 241, 7) | |
| 真值 Y | (B, 7, 161, 241, 1) | absolute SSTA |
| InitialEncoder 后 | (B, 14, 41, 61, 64) | 4x4 下采样 |
| Enc Block 0 -> mem[0] | (B, 14, 41, 61, 64) | |
| Enc Block 1 -> mem[1] | (B, 14, 21, 31, ~80) | |
| Enc Block 2 -> mem[2] | (B, 14, 11, 16, ~97) | |
| Decoder 初始 | (B, 1, 11, 16, ~97) | 单步 query |
| FinalDecoder 后 | (B, 1, 161, 241, 64) | 恢复分辨率 |
| dec_final_proj (单步) | (B, 1, 161, 241, 1) | delta SSTA |
| _ar_forward 输出 | (B, 7, 161, 241, 1) | 7步拼接, absolute SSTA |
| 掩码 mask_t | (B, 1, 161, 241, 1) | 1=海洋, 0=陆地 |

---

## 15. 配置说明

**model 关键参数**:

```yaml
data_channels: 7
input_shape:   [14, 161, 241, 7]
target_shape:  [1, 161, 241, 1]    # 单步输出, AR 7次展开
base_units: 64
scale_alpha: 0.4
enc_depth: [2, 2, 2]
dec_depth: [2, 2, 2]
num_global_vectors: 8
initial_downsample_scale: [1, 4, 4]
```

**scheduled sampling 参数** (硬编码在 __init__): 

| 参数 | 值 | 说明 |
|------|-----|------|
| ss_init | 1.0 | 初始 teacher 概率 |
| ss_final | 0.2 | 最终 teacher 概率 |
| ss_decay_epochs | 30 | 线性衰减 epoch 数 |

**损失权重** (硬编码在 _compute_loss):

| 参数 | 值 |
|------|-----|
| day_weight | [0.8, 1.5] |
| lambda_grad | 0.5 |
| lambda_tend | 5.0 |
| lambda_anti | 0.3 |

---

## 16. 训练流程

### 实验目录

```
experiments/nwp_7day/
  hparams.json
  metrics.csv              <- 断点续训追加, 不覆盖
  test_metrics.csv
  cfg.yaml
  diagnosis/               <- 诊断套件输出
  checkpoints/
    model-epoch=xxx.ckpt
    last.ckpt
    best_model.pt
```

### 运行

```bash
# 从零训练 (不能从旧 parallel-decode checkpoint 续训)
python scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py \
    --gpus 1 --save nwp_ar --data_dir datasets/SST-PREDICT/ \
    --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml

# 断点续训
... --ckpt_name last.ckpt

# 测试
... --test --ckpt_name model-epoch=XXX.ckpt

# 诊断
python scripts/cuboid_transformer/nwp_sst/diagnose_model.py \
    --exp_dir experiments/nwp_ar/ --ckpt_name model-epoch=XXX.ckpt \
    --data_dir datasets/SST-PREDICT/ \
    --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml
```

---

## 17. 模型诊断套件

10 项诊断, 3 大类:

- **I. 空间 (1.1-1.3)**: 逐像素 RMSE 热力图, 区域 RMSE (6海洋学区域), 梯度误差
- **II. 时间 (2.1-2.4)**: RMSE 逐日增长 vs persistence, 时间自相关衰减, 逐像素 persistence 对比, 逐像素时间相关
- **III. 物理 (3.1-3.3)**: dSST/dt 趋势相关, 平流一致性, 扰动传播

诊断使用独立的 `_ar_forward` 函数 (free-running, Y_teacher=None), 与 test_step 的推理路径完全一致, 确保指标可比。

---

## 18. 已知问题与改进方向

### 当前状态

| 指标 | 并行解码 | AR v2 | 改善 |
|------|:---:|:---:|:---:|
| avg RMSE | 0.483 | TBD | - |
| ACF lag-6 | 0.985 | TBD | 预期 0.75-0.85 |
| dSST/dt corr | 0.085 | TBD | 预期 0.15-0.30 |

### 已验证无效的方向
- FFT/spectral branch: 频域 != 动力学
- 并行解码 + 任意权重的 loss_tend: 梯度被架构阻断
- AR v1 (纯 teacher forcing + detach): 跨步梯度为零, SSTA 自反馈锁死

### 已验证有效的手段
- 物理通道 (grad_x, grad_y, advection): 空间结构大幅改善
- 梯度损失 (在纯 delta 上): 锋面不再被抹平
- 单步 decoder + AR 展开: 解决 parallel-decode 的因果缺失

### 潜在改进
- 将 scheduled sampling 的随机采样改为困惑度驱动 (perplexity-based)
- 用 EMA (指数移动平均) 稳定自预测路径的训练
- 增加 wind forcing 的历史天数 (当前只有最后一天)

---

## 19. 附录

| 文件 | 内容 |
|------|------|
| `src/earthformer/cuboid_transformer/cuboid_transformer.py` | 模型: CuboidAttention, Encoder, Decoder |
| `src/earthformer/datasets/nw_pacific_dataset.py` | 数据: 7通道加载, 物理派生通道 |
| `scripts/.../train_nwp_sst.py` | 训练: NWPPredictionModule, AR+Scheduled Sampling, 多物理损失 |
| `scripts/.../cfg_nwp.yaml` | 配置 |
| `scripts/.../diagnose_model.py` | 诊断: 10项实验, 独立 AR 推理 |
| `scripts/.../persistence_baseline.py` | Persistence 基线 |
| `docs/诊断分析与改进说明.md` | 诊断结果深度分析 + 改进动机 |
