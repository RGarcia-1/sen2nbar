import rioxarray  # noqa: F401
import xarray as xr

from rasterio.crs import CRS
from typing import Any
from .axioms import fiso
from .brdf import brdf
from .metadata import angles_from_metadata
from .utils import _extrapolate_c_factor


# see: https://radiantearth.github.io/stac-browser/#/external/
#                  earth-search.aws.element84.com/v1/collections/
#                  sentinel-2-l2a?.language=en
AWS_BNAMES_MAPPING = {  # Mapping Band Number to Band Name
    "B01": "coastal",
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",
    "B8A": "nir08",
    "B09": "nir09",
    "B10": "cirrus",  # NOT AVAILABLE
    "B11": "swir16",
    "B12": "swir22",
}


def c_factor(
    sun_zenith: xr.DataArray, view_zenith: xr.Dataset, relative_azimuth: xr.Dataset
) -> xr.DataArray:
    """Computes the c-factor.

    The mathematical formulation of the c-factor can be found in Equation 4 of Roy et al.,
    2008 [1]_ and Equation 7 of Roy et al., 2016 [2]_.

    Parameters
    ----------
    sun_zenith : xr.DataArray
        Sun Zenith angles in degrees.

    view_zenith : xrDataset
        Sensor Zenith angles in degrees.

    relative_azimuth : xr.Dataset
        Relative Azimuth angles in degrees.

    Returns
    -------
    xarray.DataArray
        c-factor.

    References
    ----------
    .. [1] http://dx.doi.org/10.1016/j.rse.2008.03.009
    .. [2] http://dx.doi.org/10.1016/j.rse.2016.01.023
    """
    return brdf(sun_zenith, view_zenith * 0, relative_azimuth) / brdf(
        sun_zenith, view_zenith, relative_azimuth
    )


def c_factor_from_metadata(metadata: str, is_aws: bool = False) -> xr.DataArray:
    """Gets the c-factor per band from Sentinel-2 granule metadata.

    Parameters
    ----------
    metadata : str
        Path to the metadata file. An URL can also be used.
    is_aws : bool
        Whether the STAC dataset was taken from AWS Element84.
        AWS Element84 uses different band naming convention
        from other providers.

    Returns
    -------
    xarray.DataArray
        c-factor.
    """
    # Get the available band names
    BANDS = list(fiso.keys())

    # Get the Sun and View angles
    zen_ds, azi_ds = angles_from_metadata(metadata)

    # Compute the relative azimuth per band
    relazi_ds = azi_ds["sun"] - azi_ds[BANDS]

    c = c_factor(
        sun_zenith=zen_ds["sun"], view_zenith=zen_ds[BANDS], relative_azimuth=relazi_ds
    )
    c.attrs = zen_ds.attrs
    # NOTE: c.attrs = {"crs": "EPSG:XXXXX"}

    # Drop the "band" coordinate as it creates issues later
    coords = {k: v for k, v in c.coords.items() if k != "band"}
    dims = [k for k in c.coords if k != "band"]
    c_fixed = xr.Dataset(
        data_vars={
            b: (dims, c[b].sel(band=b).data) for b in c.data_vars  # this is the worst
        },
        coords=coords,
        attrs=c.attrs
    )

    # `c_fixed` has been verified with the following:
    # >>> for b in c.data_vars:
    # >>>    aaa = c[b].sel(band=b).data
    # >>>    bbb = c_fixed[b].data
    # >>>    test = ((aaa == bbb) | (np.isnan(aaa) & np.isnan(bbb))).all()
    # >>>    print(b, test)

    # rename `c_fixed` to match the band names from AWS Element 84.
    if is_aws:
        valid_map = {
            k: v for k, v in AWS_BNAMES_MAPPING.items() if k in c_fixed.data_vars
        }
        c_fixed = c_fixed.rename_vars(valid_map)

    return c_fixed


def c_factor_from_xml(
    metadata_xml: str, dst_crs: Any, is_aws: bool = False
) -> xr.DataArray:
    """Gets the c-factor per band from a Sentinel-2 :code:`pystac.Item`.

    Parameters
    ----------
    metadata_xml : str
        metadata xml
    dst_crs : Any
        destination CRS
    is_aws : bool
        Whether the STAC dataset was taken from AWS Element84.
        AWS Element84 uses different band naming convention
        from other providers.

    Returns
    -------
    xarray.DataArray
        c-factor.
    """
    # Retrieve the EPSG from the item
    dst_crs = CRS.from_user_input(dst_crs)

    # Compute the c-factor and extrapolate
    c = c_factor_from_metadata(metadata_xml, is_aws)
    c = _extrapolate_c_factor(c)

    src_crs = CRS.from_string(c.attrs["crs"])
    c.rio.write_crs(src_crs, inplace=True)

    # If the CRSs are different: reproject
    if src_crs.to_epsg() != dst_crs.to_epsg():

        c = c.rio.reproject(dst_crs).drop("spatial_ref")
        c.rio.write_crs(dst_crs, inplace=True)

    return c
