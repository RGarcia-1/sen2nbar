import requests
import numpy as np
import numpy.typing as npt
import xarray as xr
import pandas as pd
import pystac_client
from pathlib import Path
from json import dumps as jdumps
from pystac.asset import Asset
from rasterio.crs import CRS
from requests.adapters import HTTPAdapter
from scipy.interpolate import NearestNDInterpolator
from urllib3.util import Retry


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


def _get_xml_dict(ds: xr.Dataset, stac: str, collection: str) -> dict[str, str]:
    xml_md: dict[str, str] = {}
    if "granule_metadata" in ds.coords:
        for id_, xml_ in zip(ds.id.values, ds.granule_metadata.values):
            xml_md[id_] = xml_
    else:
        # querying STAC client
        catalog = pystac_client.Client.open(stac)
        catalog_query = catalog.search(ids=ds.id.values, collections=[collection])

        items = catalog_query.item_collection()
        # NOTE: `items` do not follow the order of `da.id.values`

        # convert `items` into a pandas dataframe.
        df_items = pd.DataFrame(data={"id": [item.id for item in items], "item": items})
        df_items.set_index(keys="id", inplace=True)
        for id_ in ds.id.values:
            item = df_items.loc[id_].values[0]
            xml_md[id_] = _granule_metadata(item.assets)
    return xml_md


def _extrapolate_da(
    da: xr.DataArray,
    x_flat: npt.NDArray[np.floating] | None = None,
    y_flat: npt.NDArray[np.floating] | None = None,
) -> xr.DataArray:
    """
    Fill NaNs in an xr.DataArray using nearest-neighbor extrapolation

    Parameters
    ----------
    da : xr.DataArray
        Data array.

    x_flat, y_flat : npt.NDArray[np.floating] | None
        flattened x and y coordinate values

    Returns
    -------
    xr.DataArray
        Extrapolated data array.
    """
    data = da.data

    # Flatten the data and get all valid (non-nan) indices
    data_flat = data.ravel()
    valid_mask = ~np.isnan(data_flat)

    if not np.any(valid_mask):
        return da

    # Use precomputed meshgrid if provided
    if x_flat is None or y_flat is None:
        x, y = np.meshgrid(da.x.values, da.y.values)
        x_flat = x.ravel()
        y_flat = y.ravel()

    interpolator = NearestNDInterpolator(
        list(zip(x_flat[valid_mask], y_flat[valid_mask])), data_flat[valid_mask]
    )

    # interpolate over all pixels
    interp_vals = interpolator(x_flat, y_flat).reshape(da.data.shape)

    # only replace the nan values with `filled_values`. Here, the
    # original values are left unchanged
    return xr.DataArray(
        data=np.where(np.isnan(data), interp_vals, data),
        coords=da.coords,
        dims=da.dims,
        attrs=da.attrs,
    )


def _extrapolate_c_factor(ds: xr.Dataset) -> xr.Dataset:
    """Extrapolates the c-factor data array.

    Parameters
    ----------
    ds : xr.Dataset
        c-factor dataset

    Returns
    -------
    xr.Dataset
        Extrapolated c-factor data array.
    """

    x, y = np.meshgrid(ds.x.values, ds.y.values)
    x_flat = x.ravel()
    y_flat = y.ravel()

    return xr.Dataset(
        {
            name: _extrapolate_da(da, x_flat, y_flat)
            for name, da in ds.data_vars.items()
        },
        coords=ds.coords,
        attrs=ds.attrs,
    )


def _fetch_xml(granule_id: str, url: str, tempdir: Path, session: requests.Session):
    local_path = tempdir / f"{granule_id}.xml"
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        local_path.write_text(r.text)
    except Exception as e:
        print(f"[ERROR] Failed to fetch {granule_id}: {e}")
        local_path = None

    return granule_id, local_path


def _create_session(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    status_forcelist: list[int] = [429, 500, 502, 503, 504],
    allowed_methods: list[str] = ["HEAD", "GET", "OPTIONS"],
) -> requests.Session:
    """Creates a requests.Session with a retry strategy."""
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
