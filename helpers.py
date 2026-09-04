import numpy as np
import pandas as pd
import xarray as xr
import HydroErr as he

def stack_locations(ds, locations, var_name):
    """Nearest-neighbor extract all locations, stacked into one DataArray
    with dims (location, time)."""
    das, names = [], []
    for name, coord in locations.items():
        da = ds[var_name].sel(lat=coord["lat"], lon=coord["lon"], method="nearest")
        das.append(da.drop_vars(["lat", "lon"], errors="ignore"))
        names.append(name)
    return xr.concat(das, dim=pd.Index(names, name="location"))

def _safe_stat(sim, obs, func, min_pairs=10):
    mask = ~np.isnan(sim) & ~np.isnan(obs)
    if mask.sum() < min_pairs:
        return np.nan
    return float(func(sim[mask], obs[mask]))

def compute_monthly_metrics_groupby(pred_ds, obs_ds, locations, var_name="T", min_pairs=10):
    pred_locs = stack_locations(pred_ds, locations, var_name)
    obs_locs = stack_locations(obs_ds, locations, var_name)
    pred_locs, obs_locs = xr.align(pred_locs, obs_locs, join="inner")
    combined = xr.Dataset({"pred": pred_locs, "obs": obs_locs})

    def month_metrics(group_ds):
        kw = dict(input_core_dims=[["time"], ["time"]], vectorize=True, output_dtypes=[float])
        rmse = xr.apply_ufunc(_safe_stat, group_ds["pred"], group_ds["obs"],
                               kwargs={"func": he.rmse, "min_pairs": min_pairs}, **kw)
        r2 = xr.apply_ufunc(_safe_stat, group_ds["pred"], group_ds["obs"],
                             kwargs={"func": he.r_squared, "min_pairs": min_pairs}, **kw)
        ioa = xr.apply_ufunc(_safe_stat, group_ds["pred"], group_ds["obs"],
                              kwargs={"func": he.d, "min_pairs": min_pairs}, **kw)
        return xr.Dataset({"rmse": rmse, "r2": r2, "ioa": ioa})

    return combined.groupby("time.month").map(month_metrics)  # dims: (month, location)

def matrices_from_monthly(monthly, locations):
    """Convert the (month, location) Dataset into 3 (location x 12) numpy
    matrices, rows ordered to match your LOCATIONS dict, columns Jan->Dec."""
    loc_names = list(locations.keys())
    monthly = monthly.sortby("month").reindex(location=loc_names)
    rmse_mat = monthly["rmse"].transpose("location", "month").values

    r2_mat = monthly["r2"].transpose("location", "month").values
    ioa_mat = monthly["ioa"].transpose("location", "month").values
    return loc_names, rmse_mat, r2_mat, ioa_mat


# monthly = compute_monthly_metrics_groupby(pred_ds, obs_ds, LOCATIONS, var_name="T")
# loc_names, rmse_mat, r2_mat, ioa_mat = matrices_from_monthly(monthly, LOCATIONS)

# then straight into the same plotting function from before
#plot_metric_heatmaps(loc_names, rmse_mat, r2_mat, ioa_mat)
