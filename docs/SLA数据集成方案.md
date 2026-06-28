# SLA 数据集成方案

> 将 `/home/lab/zhangxm/tzjzllllll/sladata/slatest/` 下的每日 SLA 数据加入 Earthformer 数据集

---

## 1. SLA 数据格式评估

### 1.1 目录结构

```
/home/lab/zhangxm/tzjzllllll/sladata/slatest/
├── 2001/
│   ├── 01/
│   │   ├── 2001_01_01.nc
│   │   ├── 2001_01_02.nc
│   │   └── ...
│   └── ...
├── 2002/
└── ...  → 2025/
```

### 1.2 单个 .nc 文件预期内容（基于 CMEMS SEALEVEL_GLO_PHY_L4_MY 标准）

| 属性 | 预期值 | 说明 |
|------|--------|------|
| 变量名 | `sla` 或 `adt` | 海面高度异常 (m) |
| 空间范围 | 全球 | 需裁剪到 10°N–50°N, 120°E–180°E |
| 分辨率 | 0.25° × 0.25° | 恰好与 ERA5 分辨率一致 |
| 陆地值 | NaN | 海洋产品不含陆地数据 |
| 时间 | 单日 | 每日一个文件 |
| 数据量 | 25年 × 365天 ≈ 9125 个文件 | ~ 几个 GB |

### 1.3 必须在服务器上确认的关键信息

```bash
# 运行以下命令确认变量名、分辨率和坐标范围
python -c "
import xarray as xr
ds = xr.open_dataset('/home/lab/zhangxm/tzjzllllll/sladata/slatest/2019/01/2019_01_01.nc')
print(ds)
print('dims:', dict(ds.dims))
print('lat range:', float(ds.latitude.min()), float(ds.latitude.max()))
print('lon range:', float(ds.longitude.min()), float(ds.longitude.max()))
print('lat diff:', float(ds.latitude.diff('latitude').mean()))
print('lon diff:', float(ds.longitude.diff('longitude').mean()))
"
```

---

## 2. 数据管线设计

现有管线 vs 加入 SLA 后的管线：

```
现有:                                                  加入 SLA 后:
                                                    
SST.nc  ─→ crop+unit ─→ SST_cropped.nc              SST.nc  ─→ crop+unit ─→ SST_cropped.nc
wind.nc ─→ crop      ─→ Wind_cropped.nc             wind.nc ─→ crop      ─→ Wind_cropped.nc
                                                     sladata/ ─→ merge+crop+interp ─→ SLA_cropped.nc
                          │                                                         │
                          ▼                                                         ▼
mask.npy ← generate_ocean_mask.py                   mask.npy ← generate_ocean_mask.py
                          │                                                         │
                          ▼                                                         ▼
SST_cropped  ─→ compute_climatology  ─→ clim.nc     SST_cropped  ─→ compute_climatology  ─→ clim.nc
                          │                                                         │
                          ▼                                                         ▼
SST + clim ─→ compute_ssta ─→ ssta.nc                SST + clim ─→ compute_ssta ─→ ssta.nc
                          │                                                         │
    ┌─────────────────────┘                              ┌─────────────────────────────────────┘
    ▼                                                    ▼
build_data_array()                                  build_data_array()    
  ssta.nc + Wind_cropped.nc + mask                    ssta.nc + Wind_cropped.nc + SLA_cropped.nc + mask
    → stack([ssta, u10, v10], -1)                      → stack([ssta, u10, v10, sla], -1)
    → (T, 161, 241, 3)                                 → (T, 161, 241, 4)
```

---

## 3. 新增脚本: `preprocess_sla.py`

### 3.1 功能步骤

```
Step 1: 遍历目录树, 读取所有 yyyy/mm/yyyy_mm_dd.nc 文件
        ├── 时间范围: 2001-01-01 至 2025-12-31
        └── 按日期排序, 检查是否有缺失日期
             
Step 2: 空间裁剪至目标区域
        ├── 纬度: 10°N – 50°N  (注意 SLA 纬度可能降序)
        └── 经度: 120°E – 180°E
        
Step 3: 网格对齐 (关键步骤)
        ├── 情况A: SLA 分辨率 = 0.25° (与 ERA5 相同)
        │   ├── 经纬度 array 完全一致 → 直接使用
        │   └── 经纬度有微小偏移 → scipy.interpolate.RegularGridInterpolator 插值
        ├── 情况B: SLA 分辨率 ≠ 0.25°
        │   └── 统一插值到目标网格 (161, 241)
        └── 目标 grid:
            lat = np.linspace(10.0, 50.0, 161)  或从 SST_cropped.nc 读取
            lon = np.linspace(120.0, 180.0, 241)
            
Step 4: 陆地处理
        ├── SLA 在陆地原始为 NaN
        ├── 保持为 NaN (在 build_data_array 中统一处理)
        └── 或: 对海洋做 KNN 填充 → 先全部填充，后续掩码归零

Step 5: 保存为单个 .nc
        └── SLA_cropped.nc  (T, 161, 241), float32, units=m
```

### 3.2 伪代码

```python
#!/usr/bin/env python
"""Preprocess daily SLA data for NW Pacific SST prediction."""

import os
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

# ─── Config ───
SLA_ROOT = "/home/lab/zhangxm/tzjzllllll/sladata/slatest/"
OUTPUT_DIR = "/home/lab/zhangxm/gxy/Earthformer/datasets/SST-PREDICT/"
OUTPUT_FILE = "SLA_cropped.nc"

LAT_MIN, LAT_MAX = 10.0, 50.0
LON_MIN, LON_MAX = 120.0, 180.0

TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, 161)  # 对齐 SST_cropped
TARGET_LON = np.linspace(LON_MIN, LON_MAX, 241)

START_DATE = "2001-01-01"
END_DATE = "2025-12-31"


def load_sla_for_date(date_str):
    """Load a single SLA file."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    path = os.path.join(SLA_ROOT, f"{dt.year}/{dt.month:02d}/{date_str.replace('-', '_')}.nc")
    ds = xr.open_dataset(path)
    return ds


def crop_and_align(ds):
    """Crop to target region + interpolate to target grid."""
    # Crop lon
    ds = ds.sel(longitude=slice(LON_MIN, LON_MAX))

    # Flip lat to ascending if needed
    if ds.latitude.values[0] > ds.latitude.values[-1]:
        ds = ds.isel(latitude=slice(None, None, -1))

    # Crop lat
    ds = ds.sel(latitude=slice(LAT_MIN, LAT_MAX))

    sla_raw = ds['sla'].values  # or ds['adt']

    # Interpolate to target grid if resolution differs
    src_lat = ds.latitude.values
    src_lon = ds.longitude.values

    if src_lat.shape != (161,) or src_lon.shape != (241,):
        interp = RegularGridInterpolator(
            (src_lat, src_lon), sla_raw,
            bounds_error=False, fill_value=np.nan)
        target_pts = np.stack(np.meshgrid(TARGET_LAT, TARGET_LON, indexing='ij'), axis=-1)
        sla_aligned = interp(target_pts)
    else:
        sla_aligned = sla_raw

    return sla_aligned.astype(np.float32)


def main():
    # Build date list
    dates = []
    d = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")
    while d <= end:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    # Process all dates
    all_sla = []
    missing = []
    for date_str in tqdm(dates):
        try:
            ds = load_sla_for_date(date_str)
            sla_cropped = crop_and_align(ds)
            all_sla.append(sla_cropped)
            ds.close()
        except FileNotFoundError:
            missing.append(date_str)
            all_sla.append(np.full((161, 241), np.nan, dtype=np.float32))

    sla_data = np.stack(all_sla, axis=0)  # (T, 161, 241)
    print(f"Final shape: {sla_data.shape}")
    print(f"Missing dates: {len(missing)}")

    # Save
    time_coord = xr.cftime_range(start=START_DATE, periods=len(dates), freq='D')
    ds_out = xr.Dataset(
        {'sla': (['valid_time', 'latitude', 'longitude'], sla_data)},
        coords={
            'valid_time': time_coord,
            'latitude': TARGET_LAT,
            'longitude': TARGET_LON,
        })
    encoding = {'sla': {'dtype': 'float32', 'zlib': True, 'complevel': 4}}
    ds_out.to_netcdf(os.path.join(OUTPUT_DIR, OUTPUT_FILE), encoding=encoding)
    print(f"Saved: {OUTPUT_FILE}")
```

---

## 4. 修改现有脚本

### 4.1 `nw_pacific_dataset.py` — `compute_normalization_stats()`

新增 SLA 的海洋均值/标准差计算：

```python
def compute_normalization_stats(data_dir, train_mask=None):
    # ... 现有 ssta/u10/v10 ...
    
    # 新增 SLA
    sla_path = os.path.join(data_dir, "SLA_cropped.nc")
    ds_sla = xr.open_dataset(sla_path)
    sla = ds_sla['sla'].values
    
    # SLA: 只对海洋（mask=1 且非 NaN）计算统计
    ocean_mask = np.load(mask_path).astype(bool)
    sla_train = sla[train_mask]
    sla_ocean = sla_train[:, ocean_mask]  # (T_train, N_ocean)
    sla_valid = ~np.isnan(sla_ocean)      # 排除 NaN
    sla_mean = float(np.nanmean(sla_ocean[sla_valid]))
    sla_std = float(np.nanstd(sla_ocean[sla_valid]))
    stats['sla'] = (sla_mean, sla_std)
    
    ds_sla.close()
    return stats
```

### 4.2 `nw_pacific_dataset.py` — `build_data_array()`

```python
def build_data_array(data_dir, stats=None):
    # ... 现有加载 ssta/u10/v10 ...
    
    # 新增 SLA
    sla_path = os.path.join(data_dir, "SLA_cropped.nc")
    ds_sla = xr.open_dataset(sla_path)
    sla = ds_sla['sla'].values.astype(np.float32)  # (T, 161, 241)
    ds_sla.close()
    
    # 验证时间维度一致
    assert sla.shape[0] == ssta.shape[0], \
        f"SLA time dim {sla.shape[0]} != SSTA {ssta.shape[0]}"
    
    # 归一化 (ocean-only stats)
    sla = (sla - stats['sla'][0]) / stats['sla'][1]
    
    # 陆地归零 (同时处理 NaN 和陆地)
    sla = np.where(mask & ~np.isnan(sla), sla, 0.0)
    
    # Stack: 3通道 → 4通道
    data = np.stack([ssta, u10, v10, sla], axis=-1)  # (T, 161, 241, 4)
    
    return data, mask, years
```

### 4.3 `cfg_nwp.yaml` — 模型配置

```yaml
model:
  data_channels: 4              # 3 → 4
  input_shape: (14, 161, 241, 4)  # C=3 → 4
  # target_shape 不变 (输出仍只有 ssta)
```

### 4.4 `train_nwp_sst.py` — `_default_model()`

```python
@staticmethod
def _default_model():
    cfg = OmegaConf.create()
    cfg.data_channels = 4           # 3 → 4
    cfg.input_shape = (14, 161, 241, 4)
    # ... 其余不变 ...
```

---

## 5. 关键风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| SLA 分辨率 ≠ 0.25° | 中 | 需要插值，可能引入误差 | `RegularGridInterpolator` 做线性插值 |
| SLA 变量名非 `sla` | 中 | 脚本读取失败 | 先 `print(ds)` 确认，支持 `adt` 等别名 |
| SLA 经纬度降序 | 高 | 坐标对齐错误 | `crop_and_flip()` 复用现有逻辑 |
| 缺失日期 | 高 | 时间序列有空洞 | 记录缺失日期，填充 NaN → 掩码归零 |
| 陆地 NaN 处理 | 高 | NaN 在矩阵运算中传播 | 归一化后用 `np.where(mask & ~isnan, val, 0)` |
| SLA 值域差异大 | 确定 | 与 SSTA (σ≈0.85°C) 尺度不同 | z-score 归一化后各通道等权 |
| Initial Encoder Conv 输入通道 3→4 | 确定 | 第一层 Conv 权重重启 | 新通道对应的 kernel 随机初始化 |

---

## 6. 执行顺序

```
Step 1  [服务器]  python preprocess_sla.py
        → 生成 /home/.../datasets/SST-PREDICT/SLA_cropped.nc
        → 确认 shape = (9125, 161, 241), dtype=float32

Step 2  [本地]    修改 nw_pacific_dataset.py
        → compute_normalization_stats() 加 SLA
        → build_data_array() 加 SLA 通道

Step 3  [本地]    修改 cfg_nwp.yaml / train_nwp_sst.py  
        → data_channels: 4, input_shape: (14, 161, 241, 4)

Step 4  [服务器]  删除旧的 normalization_stats.npz
        → 首次训练时自动重新计算 (包含 SLA 统计)

Step 5  [服务器]  重新训练
        → 验证新通道是否改善精度
```

---

## 7. SLA 对模型的物理意义

SSTA 描述 SST 对气候态的偏差，SLA 描述海面高度对大地水准面的偏差。两者存在强物理耦合：

- **热膨胀效应**: SST 升高 → 海水膨胀 → SLA 升高 (相关系数 0.3–0.6，区域依赖)
- **地转流**: SLA 梯度 ∝ 表层地转流速，驱动热量平流
- **海洋涡旋**: SLA 高值对应暖涡（SSTA 高），低值对应冷涡
- **次表层信息代理**: SLA 含斜压模态信息，间接反映上层海洋热含量

加入 SLA 相当于给模型提供了**上层海洋动力状态的直接观测**，可能显著改善涡旋尺度 SSTA 预报。
