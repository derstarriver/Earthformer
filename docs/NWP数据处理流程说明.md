# 西北太平洋 SSTA 预测 — 数据处理完整流程

## 概览

```
原始 ERA5 数据 (NetCDF)
    │
    ├─ 1. 数据探查          inspect_nwp_data.py
    ├─ 2. 空间裁剪+单位转换   preprocess_nwp.py
    ├─ 3. 海陆掩码          generate_ocean_mask.py
    ├─ 4. 气候态计算         compute_climatology.py
    ├─ 5. SSTA 计算          compute_ssta.py
    ├─ 6. 训练数据集         nw_pacific_dataset.py
    └─ 7. 模型训练          train_nwp_sst.py
```

---

## 数据源

| 文件 | 变量 | 原始形状 | 时间 | 空间 |
|------|------|---------|------|------|
| `datasets/SST-PREDICT/SST.nc` | sst | (8035, 321, 561) | 2001-01~2022-12 日数据 | 100°E–240°E, 60°N–20°S, 0.25° |
| `datasets/SST-PREDICT/wind01-22.nc` | u10, v10 | 同上 | 同上 | 同上 |

---

## 步骤 1: 数据探查

**脚本**: `scripts/datasets/inspect_nwp_data.py`

**运行**:
```bash
python scripts/datasets/inspect_nwp_data.py
```

**检查内容**:
- 时间维度：首尾时间、采样频率、有无间断
- 空间维度：经纬度范围、分辨率、排序方向
- 变量统计：min/max/mean/std、缺失率、异常值
- 跨文件一致性：两文件时间/空间是否完全对齐

---

## 步骤 2: 空间裁剪 + 单位转换

**脚本**: `scripts/datasets/preprocess_nwp.py`

**运行**:
```bash
python scripts/datasets/preprocess_nwp.py
```

**输入**:
- `datasets/SST-PREDICT/SST.nc`
- `datasets/SST-PREDICT/wind01-22.nc`

**处理**:
1. 裁剪经度: 120°E–180°E
2. 翻转纬度为升序 (60°N→20°S → 10°N→50°N)
3. 裁剪纬度: 10°N–50°N
4. SST 单位转换: Kelvin → Celsius (K - 273.15)

**输出**:
- `datasets/SST-PREDICT/SST_cropped.nc` — (8035, 161, 241), °C
- `datasets/SST-PREDICT/Wind_cropped.nc` — (8035, 161, 241), m/s

---

## 步骤 3: 海陆掩码

**脚本**: `scripts/datasets/generate_ocean_mask.py`

**运行**:
```bash
python scripts/datasets/generate_ocean_mask.py
```

**原理**: SST 在陆地为 NaN, 取 t=0 时刻的 NaN 分布 → 全时间段通用掩码

**输出**:
- `datasets/SST-PREDICT/mask.npy` — (161, 241), uint8 (1=海洋, 0=陆地)
- `datasets/SST-PREDICT/mask.nc` — 带坐标信息的 NetCDF 版本
- `datasets/SST-PREDICT/mask.png` — 可视化

---

## 步骤 4: 逐日气候态

**脚本**: `scripts/datasets/compute_climatology.py`

**运行**:
```bash
python scripts/datasets/compute_climatology.py
```

**原理**: 22 年内每个日历日 (1/1...12/31) 的 SST 平均 → 366 天 × 161 × 241

**输出**:
- `datasets/SST-PREDICT/climatology.nc` — (366, 161, 241), °C
- `datasets/SST-PREDICT/annual_mean_climatology.png`
- `datasets/SST-PREDICT/summer_climatology.png`
- `datasets/SST-PREDICT/winter_climatology.png`

---

## 步骤 5: SSTA 计算

**脚本**: `scripts/datasets/compute_ssta.py`

**运行**:
```bash
python scripts/datasets/compute_ssta.py
```

**公式**:
```
SSTA(t, lat, lon) = SST(t, lat, lon) - Climatology(dayofyear(t), lat, lon)
```

**输出**:
- `datasets/SST-PREDICT/ssta.nc` — (8035, 161, 241), °C (异常值)
- `datasets/SST-PREDICT/figures/ssta_mean.png` — 时间平均 SSTA (~0)
- `datasets/SST-PREDICT/figures/ssta_std.png` — SSTA 标准差
- `datasets/SST-PREDICT/figures/sample_ssta_day.png` — 随机单日快照

---

## 步骤 6: 训练数据集

**脚本**: `src/earthformer/datasets/nw_pacific_dataset.py`

**模块入口**: `build_dataloaders(data_dir, batch_size, num_workers)`

**由训练脚本自动调用**，不需要单独运行。

**处理流程**:
1. 读取 `ssta.nc` + `Wind_cropped.nc` + `mask.npy`
2. 标准化 (先 normalize, 再 mask land→0)
   - 均值/标准差仅从训练集（2001-2020）计算
   - 保存为 `normalization_stats.npz`
3. 堆叠通道: ssta + u10 + v10 → (8035, 161, 241, 3)
4. 按年份切分:
   - 训练: 2001-2020 (7305 天 → 7289 样本)
   - 验证: 2021 (365 天 → 349 样本)
   - 测试: 2022 (365 天 → 349 样本)
5. 滑窗构造: 14 天输入 → 3 天输出
6. 返回 PyTorch DataLoader

**输出文件**:
- `datasets/SST-PREDICT/normalization_stats.npz`

---

## 步骤 7: 模型训练

**脚本**: `scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py`

**配置**: `scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml`

**运行**:
```bash
# 训练
python scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py \
    --gpus 1 --save nwp_exp1 --data_dir datasets/SST-PREDICT/ \
    --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml

# 断点续训
python scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py \
    --gpus 1 --save nwp_exp1 --data_dir datasets/SST-PREDICT/ \
    --cfg scripts/cuboid_transformer/nwp_sst/cfg_nwp.yaml --ckpt_name last.ckpt

# 测试
python scripts/cuboid_transformer/nwp_sst/train_nwp_sst.py \
    --gpus 1 --test --save nwp_exp1 --data_dir datasets/SST-PREDICT/ \
    --ckpt_name last.ckpt
```

**实验目录结构**:
```
experiments/nwp_exp1/
├── hparams.json          ← 超参数配置
├── metrics.csv           ← 每轮训练指标
├── test_metrics.csv      ← 测试分天指标 (°C)
├── cfg.yaml              ← 配置文件备份
└── checkpoints/
    ├── model-epoch=xxx.ckpt   ← 最优模型
    ├── last.ckpt              ← 最新快照
    └── best_model.pt          ← 导出权重
```

**模型输入/输出**:
```
输入: (B, 14, 161, 241, 3)  — 14d × 3通道 (ssta, u10, v10)
输出: (B, 3, 161, 241, 1)   — 3d × 1通道 (ssta)
损失: 海洋掩码加权 MSE / per-day
```

**测试指标**: 每天独立 MSE/MAE/RMSE，归一化单位 + °C

---

## 数据维度变化全览

| 阶段 | 形状 | 说明 |
|------|------|------|
| 原始 SST | (8035, 321, 561) | 100°E–240°E, 60°N–20°S |
| 原始 Wind | (8035, 321, 561, 2) | u10, v10 |
| 裁剪后 SST | (8035, 161, 241) | 120°E–180°E, 10°N–50°N, °C |
| 裁剪后 Wind | (8035, 161, 241, 2) | 同上, m/s |
| 气候态 | (366, 161, 241) | 逐日 22 年平均 |
| SSTA | (8035, 161, 241) | 异常值 |
| 合并数据 | (8035, 161, 241, 3) | ssta + u10 + v10 |
| 单样本输入 X | (14, 161, 241, 3) | 14 天历史 |
| 单样本输出 Y | (3, 161, 241, 1) | 3 天预测 |
| 批次输入 | (B, 14, 161, 241, 3) | 训练批次 |
| 掩码 | (161, 241, 1) | 海陆 |
