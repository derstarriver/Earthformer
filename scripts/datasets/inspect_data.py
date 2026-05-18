"""Inspect the structure of ENSO dataset NetCDF files."""
import xarray as xr
from pathlib import Path
import argparse
import os


def inspect_nc(filepath):
    print(f"\n{'='*60}")
    print(f"File: {filepath}")
    print(f"{'='*60}")
    if not os.path.exists(filepath):
        print("  [NOT FOUND]")
        return
    ds = xr.open_dataset(filepath)
    print(f"  Dimensions: {dict(ds.dims)}")
    print(f"  Coordinates: {list(ds.coords.keys())}")
    print(f"  Data variables: {list(ds.data_vars.keys())}")
    for var_name in ds.data_vars:
        var = ds[var_name]
        vals = var.values
        print(f"  [{var_name}]")
        print(f"    shape:  {var.shape}")
        print(f"    dims:   {list(var.dims)}")
        print(f"    dtype:  {var.dtype}")
        print(f"    min:    {vals.min():.4f}")
        print(f"    max:    {vals.max():.4f}")
        print(f"    has_nan: {bool(var.isnull().any().values)}")
    ds.close()


def main():
    parser = argparse.ArgumentParser(description="Inspect ENSO NetCDF data files")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the 4 nc files')
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print(f"Scanning directory: {data_dir}")

    for fname in ['CMIP_train.nc', 'CMIP_label.nc', 'SODA_train.nc', 'SODA_label.nc']:
        inspect_nc(data_dir / fname)

    # Summary check
    cmip_train = data_dir / 'CMIP_train.nc'
    soda_train = data_dir / 'SODA_train.nc'
    if cmip_train.exists() and soda_train.exists():
        ds1 = xr.open_dataset(cmip_train)
        ds2 = xr.open_dataset(soda_train)
        cmip_vars = set(ds1.data_vars.keys())
        soda_vars = set(ds2.data_vars.keys())
        common = cmip_vars & soda_vars
        missing_in_soda = cmip_vars - soda_vars
        missing_in_cmip = soda_vars - cmip_vars

        print(f"\n{'='*60}")
        print(f"Cross-file check")
        print(f"{'='*60}")
        print(f"  Common variables:       {common}")
        print(f"  In CMIP but not SODA:   {missing_in_soda}")
        print(f"  In SODA but not CMIP:   {missing_in_cmip}")

        # Recommend var_names based on common variables
        recommended = [v for v in ['sst', 't300', 'ua', 'va'] if v in common]
        print(f"\n  Suggested cfg.yaml var_names: {recommended}")
        print(f"  (for model input_shape, set channels = {len(recommended)})")

        ds1.close()
        ds2.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
