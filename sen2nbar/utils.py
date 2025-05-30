import numpy as np
import xarray as xr
from json import dumps as jdumps
from pystac.asset import Asset
from rasterio.crs import CRS
from scipy.interpolate import NearestNDInterpolator


class KeyErrorMessage(str):
    """
    raise KeyError() prints newlines as '\n', which causes difficulty
    reading multi-line errors. This class circumvents this.

    Usage
    -----
    msg = KeyErrorMessage("Some\nMulti-line\nError message")
    raise KeyError(msg)

    Notes
    -----
    Other python exceptions (e.g. ValueError, FileNotFoundError, TypeError)
    can handle multi-line messages.
    """

    def __repr__(self):
        return str(self)


def _get_crs(properties_md: dict) -> CRS:
    """Extract the EPSG code from the properties dictionary"""
    proj_code = properties_md.get("proj:code", None)  # EPSG:32750
    epsg_code = properties_md.get("proj:epsg", None)  # 32750

    msg1 = KeyErrorMessage(
        f"Could not find any key within the properties dict containing the CRS:\n"
        f"{jdumps(properties_md, indent=4)}"
    )

    msg2 = (
        "rasterio.crs.CRS could not convert '{0}'\n"
        "properties dict, from which the espg was extracted:\n{1}"
    )

    if epsg_code is None:
        if proj_code is None:
            raise KeyError(msg1)
        else:
            try:
                sat_crs = CRS.from_string(proj_code)
            except ValueError as e:
                emsg = msg2.format(proj_code, jdumps(properties_md, indent=4))
                raise ValueError(emsg) from e
    else:
        try:
            sat_crs = CRS.from_epsg(epsg_code)
        except ValueError as e:
            emsg = msg2.format(epsg_code, jdumps(properties_md, indent=4))
            raise ValueError(emsg) from e

    return sat_crs


def _granule_metadata(asset_md: dict[str, Asset]) -> str:
    """Extract the granule metadata from the asset dictionary"""
    gm_key = next(
        (k for k in asset_md if "granule" in k.lower() and "metadata" in k.lower()),
        None,
    )

    emsg = KeyErrorMessage(
        "Could not find any key in the assets dict containing granule metadata:\n"
        f"keys={list(asset_md.keys())}"
    )
    if gm_key is None:
        raise ValueError(emsg)

    return asset_md[gm_key].href


def _extrapolate_data_array(da: xr.DataArray) -> xr.DataArray:
    """Extrapolates a data array using Nearest Neighbor.

    Parameters
    ----------
    da : xr.DataArray
        Data array.

    Returns
    -------
    xr.DataArray
        Extrapolated data array.
    """
    # Flatten the data
    flattened = da.data.ravel()

    # Get all not nan data indices
    not_nan_idx = ~np.isnan(flattened)

    # Get the data from the flattened array
    flattened_not_nan = flattened[not_nan_idx]

    # Create a meshgrid with the coordinates values
    X, Y = np.meshgrid(da.x, da.y)

    # Flatten the meshgrid
    X = X.ravel()
    Y = Y.ravel()

    # Get the not nan coordinates
    X_not_nan = X[not_nan_idx]
    Y_not_nan = Y[not_nan_idx]

    # Initialize the interpolator
    interpolator = NearestNDInterpolator(
        list(zip(X_not_nan, Y_not_nan)), flattened_not_nan
    )

    # Do the interpolation
    Z = interpolator(X, Y)

    # Reshape the result and replace the original values
    da.values = Z.reshape(da.shape)

    return da


def _extrapolate_c_factor(da: xr.DataArray) -> xr.DataArray:
    """Extrapolates the c-factor data array.

    Parameters
    ----------
    da : xr.DataArray
        c-factor data array.

    Returns
    -------
    xr.DataArray
        Extrapolated c-factor data array.
    """

    return xr.concat(
        [_extrapolate_data_array(da.sel(band=band)) for band in da.band.values],
        dim="band",
    )
