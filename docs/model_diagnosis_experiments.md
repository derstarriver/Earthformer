# SST 模型缺陷诊断实验设计

## I. 实验总览

| # | 实验名称 | 检测目标 | 输出指标 | 回答的问题 |
|---|---------|---------|---------|-----------|
| 1.1 | 空间误差热力图 | 误差地理分布 | Per-pixel RMSE map | 模型在哪些区域失败？ |
| 1.2 | 区域分组 RMSE | 区域能力差异 | 6区 RMSE 表格 | 沿海/远洋/赤道/中纬 差异？ |
| 1.3 | 涡旋检测 accuracy | 涡旋结构保持 | Eddy detection recall | 模型是否丢失涡旋？ |
| 1.4 | 梯度误差分析 | 锋面/边界保持 | grad_error map | 模型是否模糊了锋面？ |
| 2.1 | Lead-day RMSE 曲线 | 误差增长模式 | RMSE vs lead day | 误差如何随预报天数增长？ |
| 2.2 | 自相关衰减对比 | 时间记忆能力 | ACF curve | 模型是否记住了正确的时间尺度？ |
| 2.3 | Persistence 改善率 | 相对持久性基准 | (rmse_pers - rmse_model)/rmse_pers | 模型在哪些天真正"推理"了？ |
| 2.4 | 逐格点时间相关系数 | 局部时间一致性 | Per-pixel temporal corr | 哪些格点的时序预测不可靠？ |
| 3.1 | SST 倾向诊断 | 物理输运一致性 | dSST/dt correlation | 模型预测的变温是否物理合理？ |
| 3.2 | 梯度-平流一致性 | 平流方向检验 | cos_sim(grad_SST, flow) | 模型隐式学到了平流吗？ |
| 3.3 | 扰动传播测试 | 信息传播速度 | Perturbation spread speed | 模型中信息传播速度是否物理？ |

---

## II. 各实验详细步骤

### 实验 1.1: 空间误差热力图

**数据需求**: 测试集所有样本的 pred, truth, mask

**步骤**:

1. 对每个格点 (i,j)，收集所有测试样本所有 lead day 的 (pred - truth)
2. 计算 per-pixel RMSE: `sqrt(mean((pred - truth)^2))`
3. 绘制 161×241 的 RMSE 热力图（colormap: hot）
4. 叠加大洋流标记线（黑潮：沿日本南部，亲潮：沿千岛群岛）

**如何定量分析**:
- 计算 RMSE 的空间均值、标准差
- 找出 RMSE top-5% 的格点坐标，标记在图上
- 按经纬度带统计：纬度带每 5° 一个 bin，做 RMSE vs latitude 折线图

**解释**:
- 如果沿海 RMSE >> 远洋 RMSE → 模型无法处理海岸边界不连续
- 如果黑潮延伸体 RMSE 显著高 → 模型无法处理强梯度区（mesoscale eddy 密集区）
- 如果赤道 RMSE 高 → 模型无法处理赤道波导（Kelvin/Rossby 波）

**关键图**:
- `error_map.png`: 161×241 RMSE 热力图
- `error_vs_latitude.png`: RMSE 随纬度变化
- `error_vs_longitude.png`: RMSE 随经度变化

---

### 实验 1.2: 区域分组 RMSE

**定义 6 个区域**:

| 区域 | 经度 | 纬度 | 物理特征 |
|------|------|------|---------|
| 赤道西太 | 120-150°E | 10-20°N | 暖池、弱流 |
| 赤道中太 | 150-180°E | 10-20°N | 赤道波导 |
| 黑潮区 | 130-150°E | 30-40°N | 强西边界流、强SST梯度 |
| 亲潮区 | 145-165°E | 40-50°N | 冷流、强锋面 |
| 副热带 | 150-180°E | 20-30°N | 涡旋活跃 |
| 远洋 | 160-180°E | 30-50°N | 弱流、大尺度 |

**步骤**:

1. 对每个区域创建二值 mask
2. 分别计算每个区域的 RMSE(°C) 和 MAE(°C)
3. 按 lead day 分别统计（day1/3/5/7）
4. 输出表格 + grouped bar chart

**输出表格格式**:

```
           Day1   Day3   Day5   Day7  |  Pers_Day7  Improve%
赤道西太   0.xx   0.xx   0.xx   0.xx  |  0.xx       xx%
赤道中太   ...
黑潮区     ...
...
```

**解释**:
- 黑潮区 Day7 RMSE 如果接近 Persistence → 模型对中纬度动力学没有推理能力
- 赤道区 Day7 改善率如果很高 → 模型在弱动力区靠统计模式

---

### 实验 1.3: 涡旋检测 Accuracy

**方法**: 使用 SSH/SLA-based eddy detection（AVISO 产品自带 eddy track）或基于 SST 的涡旋识别

**步骤**:

1. 使用 SLA 数据（通道3: sla）识别涡旋中心：
   - 局部极值 + 闭合等值线 → eddy mask (B, Tout, H, W)
2. 分别统计:
   - a) 涡内 RMSE: pred 在 eddy mask 内的误差
   - b) 涡外 RMSE: 海洋区域去掉 eddy mask
   - c) 涡旋检测 recall: ground truth 中有涡旋的格点是否被正确预测
3. 按涡旋属性分组:
   - 气旋式 vs 反气旋式
   - 半径 < 100km vs 100-200km vs >200km

**解释**:
- 涡内 RMSE / 涡外 RMSE > 1.5 → 模型对涡旋结构预测能力不足
- 小涡旋 recall 显著低于大涡旋 → 模型倾向于平滑小尺度结构
- 气旋/反气旋 recall 不对称 → 模型有系统性偏差

---

### 实验 1.4: 梯度误差分析

**步骤**:

1. 对 pred 和 truth 分别计算 Sobel 梯度（∂SST/∂x, ∂SST/∂y）
2. 计算梯度幅度误差: `|∇pred| - |∇truth|`
3. 计算梯度方向误差: `angle(∇pred, ∇truth)`
4. 绘制梯度误差热力图

**关键指标**:
- `grad_bias = mean(|∇pred| - |∇truth|)`: 负值 → 模型过度平滑
- `grad_angle_error = mean(acos(cos_sim))`: 锋面方向是否错误

**解释**:
- 如果 grad_bias < -0.3 (pred 梯度显著小于 truth) → 模型在消 sharp gradients（attention smoothing 效应）
- 如果 grad_angle_error > 45° → 模型学到了错误的空间梯度方向（不是物理的）
- **这是判断"图像拟合 vs 动力系统"的最直接指标**

---

### 实验 2.1: Lead-day RMSE 曲线

**步骤**:

1. 分别计算 day1~7 的 RMSE(°C)
2. 与 Persistence baseline 对比
3. 绘制: x=lead day, y=RMSE
   - 一条线: Earthformer
   - 一条线: Persistence
   - 灰色虚线: Climatology RMSE

**分析**:
- `slope = (RMSE_day7 - RMSE_day1) / 6` — **误差增长率**
- Persistence 的 slope vs Earthformer 的 slope

**解释**:
- 如果 Earthformer slope ≈ Persistence slope → 模型**没有学到动力学**，只是 day1 之后迅速退化
- 如果 Earthformer Day1 远好于 Persistence 但 Day7 接近 → 模型是短时外推器，不是动力系统模拟器
- 理想: Earthformer slope < 0.5 × Persistence slope → 模型学到了真实的时间演化

---

### 实验 2.2: 自相关衰减对比

**方法**: 对每个格点分别计算 time-lagged autocorrelation，然后空间平均

**步骤**:

1. 对每个海洋格点 (i,j):
   - a) 取 ground truth 所有测试样本的 T_out 序列
   - b) 计算 ACF_truth[i,j, lag] for lag=1..7
   - c) 取 pred 对应序列
   - d) 计算 ACF_pred[i,j, lag] for lag=1..7
2. 空间平均 (仅海洋格点):
   - ACF_truth[lag] = mean over (i,j)
   - ACF_pred[lag] = mean over (i,j)
3. 绘制: x=lag(days), y=correlation

**解释**:
- 如果 ACF_pred 衰减快于 ACF_truth → 模型**丢失了长程时间记忆**（预测随时间发散过快）
- 如果 ACF_pred[lag=7] < 0.3 而 ACF_truth[lag=7] > 0.6 → 模型时间深度不足
- 如果 ACF_pred ∼ ACF_truth 但 RMSE 仍然高 → 问题不在时间动力学，在空间精度

---

### 实验 2.3: Persistence 改善率热力图

**步骤**:

1. 对每个格点分别计算:
   ```
   improve[i,j] = (RMSE_pers[i,j] - RMSE_model[i,j]) / RMSE_pers[i,j]
   ```
2. 绘制 161×241 的 improve 热力图
   - 蓝色 (>0): 模型优于 persistence
   - 红色 (<0): 模型劣于 persistence

**预期发现**:
- 赤道西太 → 正改善（SST 变化慢，persistence 已经很强，模型难超越）
- 黑潮延伸体 → 负改善？如果模型不如 persistence → 模型在动力学活跃区退化
- 海岸边界 → 改善率最低的区域

**这是诊断"模型在哪些区域真正学了动力学"的黄金指标**

---

### 实验 2.4: 逐格点时间相关系数

**步骤**:

1. 对每个格点 (i,j)，收集所有测试样本的 (pred_dayN, truth_dayN) 序列
2. 计算 Pearson corr[i,j, N] for N=1..7
3. 绘制 corr 热力图（7个panel，或选 day1/day4/day7）

**解释**:
- corr < 0.5 的区域 → 模型的时序预测不可靠（即使 RMSE 不高也可能是统计平均的假象）
- 如果 corr 在涡旋区低但在平流区高 → 模型捕捉不到涡旋的时间演化

---

### 实验 3.1: SST 倾向诊断

**物理原理**: SST 的日变化应满足:

```
dSST/dt ≈ -u·∇SST + Q_net/(ρ·Cp·h) + κ∇²SST
         (平流项)    (热通量项)         (扩散项)
```

我们没有 Q_net 数据，但可以检验模型预测的 dSST/dt 是否与 ∇SST 模式一致。

**步骤**:

1. Ground truth dSST/dt:
   ```
   dT_truth[t] = truth[t+1] - truth[t]  (day-to-day tendency)
   ```
2. Model dSST/dt:
   ```
   dT_model[t] = pred[t+1] - pred[t]
   ```
   跨样本连接: 相邻测试样本有重叠窗口（stride=1）；取 lead day boundary: pred_day1 - truth_last_input_day
3. 空间相关系数: `corr_map = corr(dT_truth, dT_model)` over time, per pixel
4. 按 dT 大小分 bin: 强冷却 (< -0.5°C/day) / 弱变化 / 强升温 (> 0.5°C/day)，分别计算 RMSE

**解释**:
- 强升温/冷却区间 RMSE 如果显著高于弱变化区 → 模型捕捉不到强烈 SST 变化事件
- corr_map 在锋面区域如果低 → 模型不能正确预测锋面移动导致的 SST 变化

---

### 实验 3.2: 梯度-平流一致性

**方法**: 检验模型预测的 dSST/dt 是否与 SST 梯度方向一致（平流信号）

**步骤**:

1. 对每个时间步: `∇SST = [∂SST/∂x, ∂SST/∂y]`
2. 计算:
   ```
   cos_consistency = cos(angle(dSST/dt, -∇SST))
   ```
   - > 0 → 倾向与梯度反向（变冷 = 冷水平流来，物理合理）
   - < 0 → 违反平流直觉
3. 统计 cos_consistency > 0 的格点比例
4. 直接用 U10, V10 做"风驱动一致性"：风应力驱动的 SST 冷却应出现在离岸风（Ekman 抽吸）区域，检查模型预测与 truth 的一致性

**解释**:
- 如果 `cos_consistency > 0` 的比例接近 50%（随机）→ 模型没有学到平流规律
- 如果比例显著 > 50%（如 65-70%）→ 模型隐式捕获了部分平流信号
- 如果黑潮区比例特别高 → 强流区模型被迫学平流

---

### 实验 3.3: 扰动传播测试

**方法**: 给输入一个局部 SST 扰动，追踪模型输出中的扰动传播

**步骤**:

1. 选择一个测试样本 X (14, 161, 241, 4)
2. 在 t=-1 (输入最后一天) 的特定位置加一个小扰动:
   ```
   X_pert = X.clone()
   X_pert[0, -1, lat0, lon0, 0] += 1.0 (°C)
   ```
3. 分别前向传播: pred = model(X), pred_pert = model(X_pert)
4. 计算扰动传播: `delta[t, i, j] = pred_pert[t, i, j, 0] - pred[t, i, j, 0]`
5. 绘制 delta 在 day 1/3/5/7 的空间传播图

**预期行为**:
- **物理模型**: 扰动沿洋流方向传播（西南→东北，黑潮方向）、扩散衰减
- **图像拟合模型**: 扰动在全域对称衰减（像高斯模糊），或完全消失（模型忽略小扰动）
- **纯 attention 模型**: 扰动传播速度由 attention pattern 决定，可能不服从物理速度

**关键判定**:
- 如果 7 天后扰动基本消失 → 模型没有长期记忆
- 如果扰动仅局部衰减 → 模型没有传播机制，只有扩散
- 如果扰动沿特定方向传播 → 模型学到了某种定向信息流

这个实验**不需要任何代码修改** — 只需要加载训练好的模型，跑两次 inference 并做差。

---

## III. 模型缺陷决策树

```
开始
 │
 ├─ 实验 1.4: grad_bias < -0.3 且 grad_angle_error > 45°?
 │   YES → 模型是 "图像拟合器"，不是动力系统
 │         → 瓶颈: 缺乏物理输运机制
 │         → 建议: 不需要 FFT，需要 advection-aware 结构
 │         → STOP
 │   NO ↓
 │
 ├─ 实验 2.2: ACF_pred 衰减速度 >> ACF_truth?
 │   YES → 模型丢失了时间记忆
 │         → 瓶颈: 时间动力学不足
 │         → 建议: 增加 temporal context length 或改进 temporal attention
 │         → STOP
 │   NO ↓
 │
 ├─ 实验 2.3: 涡旋区 improve < 0 (模型不如 Persistence)?
 │   YES → 模型在动力学活跃区退化
 │         → 瓶颈: 模型过于保守 (smoothing)，不敢做大变化预测
 │         → 建议: loss 中加入梯度敏感项，或改进 attention 的局部分辨率
 │         → STOP
 │   NO ↓
 │
 ├─ 实验 3.1: 强 dSST (>0.5°C/day) 区间 RMSE >> 弱变化区间?
 │   YES → 模型捕捉不到极端变化事件
 │         → 瓶颈: 模型偏向均值预测
 │         → STOP
 │   NO ↓
 │
 ├─ 实验 3.2: cos_consistency > 0 的比例 ~ 50%?
 │   YES → 模型没有学到平流规律
 │         → 瓶颈: 缺乏物理输运
 │         → STOP
 │   NO ↓
 │
 ├─ 实验 1.2: 黑潮区 RMSE / 远洋区 RMSE > 2.0?
 │   YES → 模型无法处理强流、强锋面
 │         → 瓶颈: 空间分辨能力在强梯度区不足
 │         → STOP
 │   NO ↓
 │
 └─ 实验 2.1: slope_model / slope_persistence < 0.5?
     YES → 模型学到了动力学，整体性能良好
           → 如果 RMSE 仍不满意 → 问题可能在数据量或分辨率，非架构缺陷
     NO  → 模型没有持续的时间推理能力
           → 瓶颈: 模型可能是 "day1 外推器 + smooth decay"
```

---

## IV. 一句话诊断结论模板

根据以上实验，最终可以得出结论:

> **模型缺的是 [空间能力 / 时间动力学能力 / 物理输运能力]**，具体表现为 [最显著的一条实验证据]，瓶颈在 [具体的结构位置]。

例如:

> "模型缺的是**物理输运能力**，具体表现为 grad_bias = -0.42（过度平滑）且 cos_consistency ≈ 52%（平流方向随机），瓶颈在 **attention 的全局平滑效应导致锋面信息丢失**。"

例如:

> "模型缺的是**时间动力学能力**，具体表现为 ACF_pred[lag=7] = 0.28 < ACF_truth[lag=7] = 0.61，且 slope 为 Persistence 的 85%，瓶颈在 **14 天输入窗口不足以建立长期时间依赖**。"
