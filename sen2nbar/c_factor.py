import rioxarray  # noqa: F401
import numpy as np
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


def compute_relative_azimuth(vaa_ds: xr.Dataset, saa_ds) -> xr.Dataset:
    """
    The relative azimuth must range between 0 and 180 according to Roujean
    et al. (1992):
      "In these expressions [Equations (1) and (2)], theta_{s} and theta_{v}
       are the sun and view zenith angles, respectively, and phi is the rela-
       tive azimuth between sun and sensor directions, chosen by convention
       to be between zero and pi"
    The BRDF kernels introduced by Roujean et al. (1992) were adapted by
    Lucht et al. (2000)

    References
    ----------
    Roujean et al. (1992). A bidirectional reflectance model of the earth's
    surface for the correction of remote sensing data. Journal of Geophysical
    Research, 97(D18), 20,455-20,468

    Lucht et al. (2000). An Algorithm for the Retrieval of Albedo from Space
    Using Semiempirical BRDF Models. IEEE TRANSACTIONS ON GEOSCIENCE AND
    REMOTE SENSING, 38(2), 977-998
    """

    # According to:
    # https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L1C.html
    # Viewing and solar azimuths could range between 0 and 360 degrees
    delta = abs(vaa_ds - saa_ds)
    raa = xr.where(delta <= 180, delta, 360 - delta)
    return raa


def c_factor_from_metadata(
    metadata: str, y: np.ndarray, x: np.ndarray, is_aws: bool = False
) -> xr.DataArray:
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

    zen_subset_ds = zen_ds.sel(x=x, y=y, method="nearest")
    azi_subset_ds = azi_ds.sel(x=x, y=y, method="nearest")

    """
    zen_subset_ds = zen_ds.interp(
        y=y,
        x=x,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )

    azi_subset_ds = azi_ds.interp(
        y=y,
        x=x,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )
    """

    import matplotlib.pyplot as plt
    fig1, axes1 = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    axes1[0].imshow(azi_ds["B02"], interpolation="None", cmap="Greys_r")
    axes1[1].imshow(zen_ds["B02"], interpolation="None", cmap="Greys_r")
    for ax in axes1.flatten():
        ax.axis("off")

    fig2, axes2 = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    axes2[0].imshow(azi_subset_ds["B02"], interpolation="None", cmap="Greys_r")
    axes2[1].imshow(zen_subset_ds["B02"], interpolation="None", cmap="Greys_r")
    for ax in axes2.flatten():
        ax.axis("off")

    plt.show()
    exit()

    # Compute the relative azimuth per band. Note that the relative azimuth
    # must range between 0 and 180 according to Roujean et al. (1992)
    relazi_ds = compute_relative_azimuth(vaa_ds=azi_ds[BANDS], saa_ds=azi_ds["sun"])

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
    metadata_xml: str, dst_crs: Any, y: np.ndarray, x: np.ndarray, is_aws: bool = False
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
    c = c_factor_from_metadata(metadata_xml, y, x, is_aws)
    c = _extrapolate_c_factor(c)

    src_crs = CRS.from_string(c.attrs["crs"])
    c.rio.write_crs(src_crs, inplace=True)

    # If the CRSs are different: reproject
    if src_crs.to_epsg() != dst_crs.to_epsg():

        c = c.rio.reproject(dst_crs).drop("spatial_ref")
        c.rio.write_crs(dst_crs, inplace=True)

    return c
