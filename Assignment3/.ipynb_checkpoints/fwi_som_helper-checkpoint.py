"""
Functions for preparing input data for SOM, running SOM and presenting
the data.
Updated to handle NaN values to allow ocean masked data.
Added percentile based BMU plotting.

Created by: Kit Difuntorum 2025
Last edited by: Sam Walls 02/09/2026
"""

__version__ = "01.06.00"

# Required libraries

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from minisom import MiniSom
from sklearn.decomposition import PCA
from pathlib import Path

import glob
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt


def min_max_scaling(data):
    # Calculate the minimum and maximum values for each feature
    min_vals = data.min()
    max_vals = data.max()
    
    # Perform min-max scaling
    scaled_data = (data - min_vals) / (max_vals - min_vals)
    print(f'Min: {scaled_data.min().values}')
    print(f'Max: {scaled_data.max().values}')
    
    return scaled_data


def preproc_ds(da):
    # Stack lat/lon -> "space" dimension
    X = da.stack(space=('lat','lon')).transpose('time', 'space')

    # Drop grid cells (space) that are all-NaN across time
    # change how="all" to how="any", then remove fillna
    # can fix land pixels with interpolatena
    X = X.dropna(dim='space', how='all')  # (time, space)

    # Fill remaining NaNs with 0 (simple demo choice)
    X = X.fillna(0.0)

    Xz = min_max_scaling(X)
    # Xz = X

    print("Matrix shape (samples, features):", tuple(Xz.shape))
    return X, Xz


# def preproc_ds(da):
#     # Stack lat/lon -> "space" dimension
#     X = da.stack(space=('lat','lon')).transpose('time', 'space')

#     # Remove from the dataset pixels which contain any NaN values over time
#     X = X.dropna(dim='space', how='any')

#     # # Standardize each feature (column) across time
#     # mu = X.mean('time')
#     # sd = X.std('time')
#     # sd = sd.where(sd > 0, 1.0)  # avoid divide-by-zero
#     # Xz = (X - mu) / sd

#     Xz = min_max_scaling(X)
#     # Xz = X

#     print(f"Matrix shape (samples, features): {tuple(Xz.shape)}")
#     return X, Xz


def calc_som_minisom(som_x, som_y, data, num_iterations):
   
    som = MiniSom(
        x=som_x, y=som_y,
        input_len=data.shape[1],
        sigma=2.0,          # neighborhood: how big of an area do you look for other vectors to 'drag in'
        learning_rate=0.5,  #
        neighborhood_function='gaussian',
        random_seed=0
    )

    som.random_weights_init(data)
    som.train_random(data, num_iteration=num_iterations, verbose=True)
    #som.train_batch()

    print(f"Trained SOM: {som_x}x{som_y} with {num_iterations} iterations.")
    
    return som


def bmu_assign(som, som_y, X):
    # BMUs -> labels 0..(som_x*som_y-1)
    bmus = [som.winner(s) for s in X.values]           # list of (i, j)
    labels = np.array([i * som_y + j for (i, j) in bmus])

    # Attach to time for convenience
    som_labels = xr.DataArray(labels, coords={'time': X['time']}, dims=['time'], name='som_label')
    # display(som_labels.to_pandas().head())

    # Frequency table
    unique, counts = np.unique(labels, return_counts=True)
    freq = dict(zip(unique, counts))
    # print('Cluster counts (unit_id: count):')
    # print(freq)
    
    return som_labels, unique, counts, freq


def sort_labels(som_labels, som_x, som_y) -> dict:
    """
    Sorts SOM timesteps into categories for each unique unit
    label.
    """

    labels_dict = {key: [] for key in range(som_x * som_y)}
    for label in som_labels:
        key = int(label.values)
        entry = label.time.values
        labels_dict[key].append(entry)

    return labels_dict

    
# def create_proto_maps(som, som_x, som_y, X_orig):
#     """
#     Convert scaled MiniSom weights back to FWI maps.

#     X_orig is the original unscaled stacked array returned by preproc_ds().
#     """

#     # Scaled SOM weights
#     W_scaled = som.get_weights().reshape(
#         som_x * som_y,
#         X_orig.shape[1]
#     )

#     # Global FWI range used for min-max scaling
#     fwi_min = float(X_orig.min())
#     fwi_max = float(X_orig.max())

#     # Undo global min-max scaling
#     W_original = W_scaled * (fwi_max - fwi_min) + fwi_min

#     # Restore the space MultiIndex
#     proto_maps = xr.DataArray(
#         W_original,
#         coords={
#             "unit": np.arange(som_x * som_y),
#             "space": X_orig["space"]
#         },
#         dims=("unit", "space"),
#         name="prototype"
#     )

#     # Restore lat/lon and sort for plotting
#     proto_maps = (
#         proto_maps
#         .unstack("space")
#         .sortby(["lat", "lon"])
#     )

#     return proto_maps


def plot_field(ax, field2d, title=""):
    """
    field2d: DataArray (lat, lon)
    Draws pcolormesh, coastlines, and masks oceans to white.
    """
    proj = ccrs.PlateCarree()
    ax.set_title(title)
    # Plot the data
    im = ax.pcolormesh(field2d['lon'], field2d['lat'], field2d,
                       transform=proj, shading='auto')
    # Coastlines
    ax.coastlines(linewidth=0.6, color='black', zorder=50)
    # Mask oceans by drawing OCEAN feature on top (white fill)
    ax.add_feature(cfeature.OCEAN, facecolor='white', edgecolor='none', zorder=60)
    # Optional: thin land outline too (helps at coast)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=70)
    # Tight extent to the data grid
    ax.set_extent([float(field2d['lon'].min()), float(field2d['lon'].max()),
                   float(field2d['lat'].min()), float(field2d['lat'].max())],
                   crs=proj)
    return im


def create_percentile_maps(
    X_orig,
    som_labels,
    som_x,
    som_y,
    percentile=90,
):
    """
    Create pixel-wise percentile maps for each SOM unit.

    For each SOM unit:
        1. Select all time steps assigned to that BMU.
        2. Calculate the requested percentile at each spatial pixel.
        3. Reconstruct the result as a lat/lon map.

    Parameters
    ----------
    X_orig : xarray.DataArray
        Original unscaled data with dimensions ('time', 'space').

    som_labels : xarray.DataArray
        BMU assignment for each time step.

    som_x, som_y : int
        Dimensions of SOM grid.

    percentile : float
        Percentile to calculate, e.g. 90 for the 90th percentile.

    Returns
    -------
    xarray.DataArray
        Percentile maps with dimensions ('unit', 'lat', 'lon').
    """

    q = percentile / 100.0
    n_units = som_x * som_y

    maps = []

    for unit_id in range(n_units):

        # Select all time steps belonging to this BMU
        members = X_orig.where(
            som_labels == unit_id,
            drop=True
        )

        if members.sizes["time"] == 0:
            # Handle empty SOM units
            field = xr.full_like(
                X_orig.isel(time=0, drop=True),
                np.nan
            )
        else:
            # Pixel-wise percentile across all times in this BMU
            field = members.quantile(
                q,
                dim="time"
            )

        # Remove scalar quantile coordinate if present
        if "quantile" in field.coords:
            field = field.drop_vars("quantile")

        field = field.expand_dims(unit=[unit_id])

        maps.append(field)

    percentile_maps = xr.concat(
        maps,
        dim="unit"
    )

    # Convert stacked space dimension back to lat/lon
    percentile_maps = (
        percentile_maps
        .unstack("space")
        .sortby("lat")
        .sortby("lon")
    )

    percentile_maps.name = f"FWI_p{percentile:g}"

    percentile_maps.attrs["long_name"] = (
        f"{percentile:g}th percentile FWI by SOM unit"
    )

    return percentile_maps


def create_proto_maps(som, som_x, som_y, X_orig):
    """
    Convert MiniSom prototype weights from scaled feature space back into
    FWI units and reconstruct them as latitude-longitude maps.

    This assumes the SOM was trained using global min-max scaling:

        X_scaled = (X_orig - X_orig.min()) / (
            X_orig.max() - X_orig.min()
        )

    Parameters
    ----------
    som : MiniSom
        Trained MiniSom object.

    som_x, som_y : int
        Dimensions of the SOM grid.

    X_orig : xarray.DataArray
        Original unscaled data with dimensions ('time', 'space').
        The 'space' coordinate must be the MultiIndex created by stacking
        latitude and longitude.

    Returns
    -------
    xarray.DataArray
        Prototype maps with dimensions ('unit', 'lat', 'lon'), expressed
        in the original FWI units.
    """

    n_units = som_x * som_y
    n_features = X_orig.sizes["space"]

    # MiniSom weights have shape:
    # (som_x, som_y, n_features)
    W_scaled = som.get_weights()

    expected_shape = (som_x, som_y, n_features)

    if W_scaled.shape != expected_shape:
        raise ValueError(
            f"Unexpected SOM weight shape {W_scaled.shape}. "
            f"Expected {expected_shape}."
        )

    # Flatten the SOM grid into a single unit dimension
    W_scaled = W_scaled.reshape(n_units, n_features)

    # Recover the global range used during min-max scaling
    fwi_min = float(X_orig.min().compute())
    fwi_max = float(X_orig.max().compute())

    fwi_range = fwi_max - fwi_min

    if not np.isfinite(fwi_range) or fwi_range <= 0:
        raise ValueError(
            f"Invalid FWI scaling range: min={fwi_min}, max={fwi_max}"
        )

    # Undo global min-max scaling
    W_original = W_scaled * fwi_range + fwi_min

    # Attach the original stacked spatial coordinate
    proto_maps = xr.DataArray(
        W_original,
        coords={
            "unit": np.arange(n_units),
            "space": X_orig["space"],
        },
        dims=("unit", "space"),
        name="prototype",
        attrs={
            "long_name": "SOM prototype FWI",
            "units": X_orig.attrs.get("units", ""),
        },
    )

    # Restore the 2D spatial grid and ensure coordinates are monotonic
    proto_maps = (
        proto_maps
        .unstack("space")
        .sortby("lat")
        .sortby("lon")
    )

    return proto_maps


def plot_som_maps(
    proto_maps,
    som_x,
    som_y,
    min_value=None,
    max_value=None,
    nz_extent=(165, 180, -48.8, -33.5),
    cmap="gist_rainbow"
):
    """
    Plot reconstructed SOM prototype maps using a shared colour scale.

    Parameters
    ----------
    proto_maps : xarray.DataArray
        Prototype maps with dimensions ('unit', 'lat', 'lon').

    som_x, som_y : int
        Dimensions of the SOM grid.

    min_value, max_value : float or None
        Shared colour limits. If omitted, the minimum and maximum across
        all prototype maps are used.

    nz_extent : tuple
        Map extent as (west, east, south, north).
    """

    required_dims = {"unit", "lat", "lon"}

    if not required_dims.issubset(proto_maps.dims):
        raise ValueError(
            "proto_maps must contain the dimensions "
            "'unit', 'lat', and 'lon'."
        )

    expected_units = som_x * som_y

    if proto_maps.sizes["unit"] != expected_units:
        raise ValueError(
            f"proto_maps contains {proto_maps.sizes['unit']} units, "
            f"but som_x * som_y is {expected_units}."
        )

    # Ensure pcolormesh receives monotonically ordered coordinates
    proto_maps = (
        proto_maps
        .sortby("lat")
        .sortby("lon")
    )

    lat_increasing = np.all(np.diff(proto_maps["lat"].values) > 0)
    lon_increasing = np.all(np.diff(proto_maps["lon"].values) > 0)

    if not lat_increasing or not lon_increasing:
        raise ValueError(
            "Latitude and longitude coordinates must be strictly "
            "increasing after sorting."
        )

    if min_value is None:
        min_value = float(proto_maps.min().compute())

    if max_value is None:
        max_value = float(proto_maps.max().compute())

    if min_value >= max_value:
        raise ValueError(
            f"min_value must be below max_value, got "
            f"{min_value} and {max_value}."
        )

    fig, axes = plt.subplots(
        nrows=som_x,
        ncols=som_y,
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(12, 10),
    )

    # Ensures axes[i, j] works for 1-row or 1-column SOMs
    axes = np.asarray(axes).reshape(som_x, som_y)

    im = None

    for i in range(som_x):
        for j in range(som_y):
            unit_id = i * som_y + j
            ax = axes[i, j]

            prototype = proto_maps.isel(unit=unit_id)

            im = ax.pcolormesh(
                prototype["lon"].values,
                prototype["lat"].values,
                prototype.values,
                transform=ccrs.PlateCarree(),
                shading="auto",
                cmap=cmap,
                vmin=min_value,
                vmax=max_value,
            )

            ax.add_feature(
                cfeature.OCEAN,
                facecolor="white",
                edgecolor="none",
                zorder=60,
            )

            ax.coastlines(
                linewidth=0.6,
                color="black",
                zorder=70,
            )

            ax.set_extent(
                nz_extent,
                crs=ccrs.PlateCarree(),
            )

            ax.set_title(
                f"Unit {unit_id} ({i},{j})",
                fontsize=10,
            )

    cbar = fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.05,
        pad=0.06,
    )

    cbar.set_label("FWI")

    plt.show()
    
def som_distance_map(som):
    u = som.distance_map()  # shape (som_x, som_y)
    plt.figure(figsize=(5, 4))
    plt.imshow(u, origin='upper', cmap='coolwarm')
    plt.colorbar(label='(avg neighbor distance)')
    plt.title('SOM Distance Map')
    plt.xlabel('j (x-axis of SOM grid)')
    plt.ylabel('i (y-axis of SOM grid)')
    plt.tight_layout()
    plt.grid(False)
    plt.show()
    
def som_distribution(som, som_x, som_y, data):
    # BMUs -> labels
    bmus = [som.winner(s) for s in data]  
    labels = np.array([i * som_y + j for (i, j) in bmus])

    # Simple bar chart
    counts = np.bincount(labels, minlength=som_x * som_y)
    plt.figure(figsize=(6, 3.5))
    plt.bar(np.arange(som_x * som_y), counts)
    plt.xlabel('SOM Unit ID (0 to 8)')
    plt.ylabel('Sample count')
    plt.title('Samples per Cluster')
    plt.tight_layout()
    plt.show()

    # Also print a small table
    for k, c in enumerate(counts):
        print(f'Unit {k:>2}: {c} samples')
        
    return labels


def plot_with_coasts_no_ocean(
    da2d: xr.DataArray,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str | None = None,
    coastline_res: str = "10m",
    title: str | None = None,
    figsize=(7, 7),
):
    """
    Plot a 2D (lat, lon) DataArray with coastlines and oceans masked to white.

    Parameters
    ----------
    da2d : xr.DataArray
        2D data with latitude/longitude coordinates. If your data has a time
        dimension, pass a single time slice (e.g., da.sel(time='2010-01-08')).
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float or None
        Color scale limits. If None, inferred from the data.
    cbar_label : str or None
        Colorbar label.
    coastline_res : {'10m','50m','110m'}
        Cartopy coastline resolution.
    title : str or None
        Plot title.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    fig, ax
    """

    # --- Find lat/lon coord names
    lat_name = None
    lon_name = None
    for cand in ("lat", "latitude", "y"):
        if cand in da2d.coords:
            lat_name = cand
            break
    for cand in ("lon", "longitude", "x"):
        if cand in da2d.coords:
            lon_name = cand
            break
    if lat_name is None or lon_name is None:
        raise ValueError("Could not find latitude/longitude coords in DataArray.")

    if da2d.ndim != 2:
        # If it has extra dims (e.g., time), ask user to select one
        extra = [d for d in da2d.dims if d not in (lat_name, lon_name)]
        raise ValueError(
            f"Expected 2D DataArray (lat, lon). Found dims {da2d.dims}. "
            f"Select a slice first (e.g., da.sel(time=...)). Extra dims: {extra}"
        )

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": proj})
    nz_extent = [165, 180, -48.8, -33.5]

    # Plot the data
    im = da2d.plot(
        ax=ax,
        transform=proj,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=True,
        cbar_kwargs={"label": cbar_label} if cbar_label else None,
    )

    # Mask oceans visually (draw ocean polygons on top, filled white)
    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=10)

    # Coastlines on top
    ax.coastlines(resolution=coastline_res, linewidth=0.8, zorder=20)

    # Tighten extent to the data grid
    ax.set_extent(nz_extent, crs=ccrs.PlateCarree())
    
    # ax.set_extent(
    #     [
    #         float(da2d[lon_name].min()),
    #         float(da2d[lon_name].max()),
    #         float(da2d[lat_name].min()),
    #         float(da2d[lat_name].max()),
    #     ],
    #     crs=proj,
    # )

    if title:
        ax.set_title(title)

    return fig, ax

    