# 西北太平洋 SST 预测 — Earthformer 适配方案

> 目标：0.25°×0.25° 分辨率，预测西北太平洋海表温度

---

## 1. 任务变化总览

| 项目 | 当前 ENSO 任务 | 西北太平洋 SST 任务 |
|------|---------------|-------------------|
| 核心目标 | 预测 Niño 3.4 指数 | 逐像素预测 SST |
| 空间范围 | 全球 95°E–330°E, 55°S–60°N | 西北太平洋 ~100–180°E, 0–50°N |
| 网格尺寸 | 24×48 (低分辨率) | ~200×320 (0.25°, 估计值) |
| 通道数 | 4 (sst, t300, ua, va) | ≥1 (至少 sst，可加入辅助变量) |
| 评估指标 | Niño 相关系数 / RMSE | 像素级 SST MSE/MAE/RMSE、空间相关系数 |
| 模型深度 | 2 层编码器 | 3–4 层（网格大了很多） |

---

## 2. 数据准备

### 2.1 数据源

建议使用以下再分析/观测数据：

| 数据源 | 分辨率 | 时段 | 变量 |
|--------|--------|------|------|
| **ERA5** | 0.25° | 1940–现在 | sst, u10, v10, msl 等 |
| **OISST** | 0.25° | 1981–现在 | 仅 SST (卫星观测) |
| **SODA** | 0.25° | 1980–现在 | sst, t300, ua, va 等海洋变量 |

### 2.2 数据裁剪

过滤到西北太平洋范围：

```python
# 建议范围
lat_range = (0.0, 50.0)    # 0°N – 50°N
lon_range = (100.0, 180.0) # 100°E – 180°E

# 0.25° 分辨率下网格点数
n_lat = int((50.0 - 0.0) / 0.25) + 1  = 201
n_lon = int((180.0 - 100.0) / 0.25) + 1 = 321
# 网格: (201, 321) ≈ 64,500 空间点
```

### 2.3 NetCDF 格式转换

将你的 0.25° 数据转换为与当前代码兼容的 NetCDF 格式：

```python
import xarray as xr
import numpy as np

# 假设你有 monthly SST 数据，形状 (years, months, lat, lon)
# 转换为 Earthformer 兼容格式 (year, 36, lat, lon)
# 36 = 3年 × 12月 (与原始 ICAR-ENSO 格式兼容)

ds = xr.Dataset({
    'sst': (['year', 'month', 'lat', 'lon'], sst_data),
    # 可选: 加入辅助变量
    # 'ssh': (['year', 'month', 'lat', 'lon'], ssh_data),
    # 'curl': (['year', 'month', 'lat', 'lon'], wind_curl_data),
}, coords={
    'year': np.arange(n_years),
    'month': np.arange(36),  # 每行36个月 (3个日历年)
    'lat': lat_values,
    'lon': lon_values,
})
ds.to_netcdf('NWP_train.nc')
```

**关键要点**：
- 变量名用 `sst`（与当前代码默认变量列表兼容）
- 坐标命名: `year`, `month`, `lat`, `lon`
- month 维度必须是 36（即使你的数据是 12 月/年，也要压成 36）

---

## 3. 代码修改清单

### 3.1 数据预处理脚本

**文件**: `scripts/datasets/preprocess_enso.py`

修改内容：
```python
# 1. 去掉经度过滤（数据已裁剪到西北太平洋）
# 注释掉: lon_mask = np.logical_and(lon_vals >= 95, lon_vals <= 330)

# 2. 只保留 SST 通道（如果只有 SST）
DEFAULT_VARS = ['sst']  # 而不是 ['sst', 't300', 'ua', 'va']

# 3. 调整 CMIP 数据拆分配置（如果使用不同数据集）
cmip6_cutoff = <你的训练数据的年份行数>
```

### 3.2 训练配置文件

**文件**: `scripts/cuboid_transformer/enso/cfg.yaml`

```yaml
dataset:
  in_len: 12
  out_len: 12              # 改短: 预测1年 (西北太平洋SST可预测性较短)
  in_stride: 1
  out_stride: 1
  train_samples_gap: 1
  eval_samples_gap: 1
  cmip6_cutoff: <你的训练数据年份行数>
  cmip6_years_per_model: <每个模式年份数>
  soda_val_ratio: 0.2      # 用20%做验证
  var_names: ["sst"]       # 单变量或 ["sst", "ssh", "curl"]

layout:
  in_len: 12
  out_len: 12
  layout: "NTHWC"

model:
  data_channels: 1         # SST 单通道
  input_shape: [12, 201, 321, 1]   # 根据实际网格调整
  target_shape: [12, 201, 321, 1]
  base_units: 64
  enc_depth: [2, 2, 2]    # 3层编码器 (原2层→3层)
  dec_depth: [2, 2, 2]    # 3层解码器
  downsample: 2            # 每层 H/W 减半

  # 201×321 → 100×160 → 50×80 → 25×40 (3次下采样后)

  # 注意力策略: 逐层不同
  self_pattern: ["axial", "spatial_lg_4", "divided_st"]
  cross_self_pattern: ["axial", "spatial_lg_4", "divided_st"]
  cross_pattern: ["cross_1x1", "cross_1x1", "cross_1x1"]

  # 全局向量
  num_global_vectors: 8    # 开启!
  use_dec_self_global: true
  use_dec_cross_global: true
  use_global_vector_ffn: true

  initial_downsample_type: "conv"
  initial_downsample_scale: [1, 2, 2]  # 初始就做2倍下采样
  initial_downsample_conv_layers: 2
  final_upsample_conv_layers: 2

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

optim:
  total_batch_size: 32
  micro_batch_size: 2      # 网格变大 → batch 变小
  method: "adamw"
  lr: 0.0001
  wd: 1.0e-05
  max_epochs: 100
  lr_scheduler_mode: "cosine"
  warmup_percentage: 0.2
  min_lr_ratio: 1.0e-3
  early_stop: true
  early_stop_mode: "min"
  early_stop_patience: 10
  save_top_k: 3
```

### 3.3 训练脚本

**文件**: `scripts/cuboid_transformer/enso/train_cuboid_enso.py`

关键修改点：

```python
# 1. 移除 Niño 相关代码 (西北太平洋没有 Niño 3.4)
# 删除 sst_to_nino() 调用
# 删除 compute_enso_score() 调用
# 删除 nino_preds / nino_target 的收集和计算

# 2. 修改 validation_step
def validation_step(self, batch, batch_idx, dataloader_idx=0):
    pred_seq, loss, in_seq, target_seq, _ = self(batch)
    # 只保留 SST MSE/MAE
    self.valid_mse(pred_seq, target_seq)
    self.valid_mae(pred_seq, target_seq)
    return loss

# 3. 修改 forward (如果只有 SST)
def forward(self, batch):
    data_seq, _ = batch              # 不再有 nino_target
    data_seq = data_seq.float()
    in_seq = data_seq[:, :12, ...]
    target_seq = data_seq[:, 12:24, ...]
    pred_seq = self.torch_nn_module(in_seq)
    loss = F.mse_loss(pred_seq, target_seq)
    return pred_seq, loss, in_seq, target_seq, None

# 4. ModelCheckpoint 监控 SST MSE
monitor="valid_mse_epoch"  # 不是 valid_sst_mse_epoch
```

### 3.4 数据加载器

**文件**: `src/earthformer/datasets/enso/enso_dataloader.py`

修改内容：
```python
# 1. 去掉 Niño 3.4 相关函数
# 注释掉: find_nino_indices, compute_nino_from_sst

# 2. 去掉经度过滤 (数据已裁剪到目标区域)
# read_cmip_multivar / read_soda_multivar 中注释掉:
# lon_mask = np.logical_and(lon_vals >= 95, lon_vals <= 330)
# ds_train = ds_train.sel(lon=lon_mask)

# 3. Dataset.__getitem__ 只返回数据 (不再包含 nino_target)
def __getitem__(self, idx):
    seq_idx = self.idx_seq[idx]
    x = self.data[seq_idx].copy()
    return torch.from_numpy(x), torch.zeros(1)  # dummy label

# 4. 或者自建 NWP 专用数据集类
```

### 3.5 评估指标模块

**文件**: `src/earthformer/metrics/enso.py`

可选择性移除 Niño 相关函数 (`compute_enso_score`, `sst_to_nino`)，或保留备用。新增：

```python
def compute_spatial_corr(pred, target):
    """计算预测与真值的空间相关系数"""
    pred_anom = pred - pred.mean(dim=(2,3), keepdim=True)
    target_anom = target - target.mean(dim=(2,3), keepdim=True)
    corr = (pred_anom * target_anom).sum(dim=(2,3)) / (
        torch.sqrt((pred_anom**2).sum(dim=(2,3)) * (target_anom**2).sum(dim=(2,3))) + 1e-8)
    return corr.mean()
```

---

## 4. 模型架构对比

### 当前 ENSO 架构 (浅)
```
输入: (12, 24, 48, 4)
  → Conv → (12, 24, 24, 64)
  → Enc + axial [24×24] → (12, 24, 24, 64)
  → PatchMerge → (12, 12, 12, 128)
  → Enc + axial [12×12] → (12, 12, 12, 128)
  → Dec → (26, 24, 48, 4)
```

### 西北太平洋架构 (深)
```
输入: (12, 201, 321, 1)
  → Conv → (12, 100, 160, 64)       # initial_downsample [1,2,2]
  → Enc + axial [100×160] → mem[0]  # 全局大尺度
  → PatchMerge → (12, 50, 80, 128)
  → Enc + spatial_lg_4 [50×80] → mem[1]  # 局部纹理
  → PatchMerge → (12, 25, 40, 256)
  → Enc + divided_st [25×40] → mem[2]    # 时空分离
  → Dec + global_vectors → (12, 201, 321, 1)
```

**计算量估算**:
- 当前: 1.4M 参数, ~170 TFLOPs/epoch (在 24×48 网格)
- 新模型: ~3-5M 参数, ~500-1000 TFLOPs/epoch (在 201×321 网格)
- 单卡 A6000: 每 epoch ~5-10 分钟 → 100 epoch ≈ 8-16 小时

---

## 5. 评估指标

西北太平洋 SST 预测应使用以下指标：

| 指标 | 公式 | 含义 |
|------|------|------|
| **全域 RMSE** | sqrt(mean((pred-true)²)) | 整体预测精度 |
| **全域 MAE** | mean(\|pred-true\|) | 平均绝对误差 |
| **空间相关系数** | corr(pred_map, true_map) | 空间模式相似度 |
| **区域 RMSE** | 分区域（黑潮区、沿岸区等） | 区域精度 |
| **持续性基线** | persist = MSE(clim, true) | 模型是否优于气候态 |
| **RMSE 比持续性** | rmse_model / rmse_persist | <1 表示有技巧 |

---

## 6. 实施步骤

### 第一步：数据准备 (1-2天)
1. 确认数据源（ERA5 / SODA / 其他 0.25° 数据）
2. 裁剪到西北太平洋区域
3. 转换为 year×36 month×lat×lon 的 NetCDF 格式
4. 划分训练/验证/测试集

### 第二步：预处理 (一次性)
1. 修改 `preprocess_enso.py`（变量列表、经纬度过滤）
2. 运行一次生成 `.npz` 缓存

### 第三步：配置修改 (一次性)
1. 更新 `cfg.yaml`（网格尺寸、层数、注意力策略、batch size）
2. 更新训练脚本（去掉 Niño 相关代码、修改 loss/metrics）

### 第四步：小规模测试 (1小时)
1. `max_epochs=3` 验证模型能跑通
2. 确认显存够用（A6000 48G 应该够）

### 第五步：正式训练 (1天)
1. `max_epochs=100`、开启 `early_stop`
2. 监控 `valid_mse_epoch`

---

## 7. 风险与建议

| 风险 | 应对 |
|------|------|
| 显存不足 | micro_batch_size 降到 1, 增加 accumulate_grad_batches |
| 201×321 网格太大 | 先用 initial_downsample_scale=[1,4,4] 或先用双线性插值缩到 100×160 |
| 训练太慢 | enc_depth 降为 [1,1,1], 或先只用前2层试 |
| 缺少 t300/ua/va | 单 SST 也能训练, 加入 SSH 或风场可提升效果 |
| 西北太平洋 SST 可预测性低 | out_len 从 26 缩到 6-12, 月尺度可预测性远不如 ENSO |
