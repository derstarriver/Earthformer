# Earthformer ENSO/SST 多变量预测教程

## 1. 数据放置

将 4 个 NetCDF 文件放到 `datasets/enso_multivar/`：

```bash
mkdir -p datasets/enso_multivar
# 放入文件: CMIP_train.nc, CMIP_label.nc, SODA_train.nc, SODA_label.nc
```

### 数据格式要求

| 文件 | 维度 | 变量 |
|------|------|------|
| `CMIP_train.nc` | (year, month, lat, lon) | sst, t300, ua, va |
| `CMIP_label.nc` | (year, month) | nino |
| `SODA_train.nc` | (year, month, lat, lon) | sst, t300, ua, va |
| `SODA_label.nc` | (year, month) | nino |

## 2. 检查数据

```bash
python scripts/datasets/inspect_data.py --data_dir ./datasets/enso_multivar/
```

网格维度和 Niño 3.4 区域索引会在运行时**自动检测**，无需手动配置。

## 3. 训练

```bash
# 单 GPU
python scripts/cuboid_transformer/enso/train_cuboid_enso.py \
    --gpus 1 --save enso_exp1 \
    --data_dir ./datasets/enso_multivar/

# 多 GPU 训练
MASTER_ADDR=localhost MASTER_PORT=10001 \
python scripts/cuboid_transformer/enso/train_cuboid_enso.py \
    --gpus 2 --cfg scripts/cuboid_transformer/enso/cfg.yaml \
    --save enso_exp1 --data_dir ./datasets/enso_multivar/

# 从 checkpoint 恢复
python scripts/cuboid_transformer/enso/train_cuboid_enso.py \
    --gpus 2 --cfg scripts/cuboid_transformer/enso/cfg.yaml \
    --ckpt_name last.ckpt --save enso_exp1 --data_dir ./datasets/enso_multivar/
```

## 4. 测试

```bash
python scripts/cuboid_transformer/enso/train_cuboid_enso.py \
    --gpus 1 --test --ckpt_name best.ckpt \
    --save enso_exp1 --data_dir ./datasets/enso_multivar/
```

## 5. 模型说明

- **输入**: 12 月 × 4 通道 (sst, t300, ua, va) × 自适应网格
- **输出**: 26 月 × 4 通道预测
- **主要指标**: SST MSE/MAE（SST 通道）
- **辅助指标**: Niño 3.4 相关系数/RMSE（自动从坐标定位区域）
- **训练数据**: CMIP6 + CMIP5 模式
- **验证/测试**: SODA 观测数据（按比例分割）

## 6. 关键参数

在 `cfg.yaml` 中可调整：

| 参数 | 默认 | 说明 |
|------|------|------|
| `dataset.cmip6_cutoff` | 2265 | CMIP6 前 N 个 year 行 |
| `dataset.cmip6_years_per_model` | 151 | 每个 CMIP6 模式的 year 行数 |
| `dataset.soda_val_ratio` | 0.5 | SODA 验证集比例 |
| `optim.lr` | 1e-4 | 学习率 |
| `optim.max_epochs` | 100 | 最大 epoch |
