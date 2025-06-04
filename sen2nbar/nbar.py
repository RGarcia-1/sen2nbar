import os
import tempfile
from glob import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
import rioxarray
import xarray as xr

from .axioms import bands
from .c_factor import c_factor_from_xml, c_factor_from_metadata
from .metadata import get_processing_baseline
from .utils import _extrapolate_c_factor
from .utils import _get_xml_dict
from .utils import _fetch_xml
from .utils import _create_session
from .utils import _apply_scale_offset


OTHER_DVARS = ["aot", "scl", "wvp"]


def nbar_safe(
    path: str | Path, cog: bool = True, to_int: bool = False, quiet: bool = False
) -> None:
    """Computes the Nadir BRDF Adjusted Reflectance (NBAR) using the SAFE path.

    If the processing baseline is greater than 04.00, the DN values are automatically
    shifted before computing NBAR. All images are saved in the SAFE path inside the folder
    NBAR.

    Parameters
    ----------
    path : str | Path
        SAFE path.
    cog : bool, default = True
        Whether to save the images as Cloud Optimized GeoTIFF (COG).
    to_int : bool, default = False
        Whether to convert the NBAR output to integer.
    quiet : bool, default = False
        Whether to show progress.

    Returns
    -------
    None
    """
    path = Path(path)
    # Whether to save as COG
    if cog:
        driver = "COG"
    else:
        driver = None

    # NBAR folder to store the images
    nbar_output_path = path / "NBAR"
    nbar_output_path.mkdir(exist_ok=True)

    # Metadata file
    metadata = glob(os.path.join(path, "GRANULE", "*", "MTD_TL.xml"))[0]

    # Processing baseline
    PROCESSING_BASELINE = get_processing_baseline(path / "MTD_MSIL2A.xml")

    # Whether to shift the DN values
    # After 04.00 all DN values are shifted by 1000
    harmonize = PROCESSING_BASELINE >= 4.0

    # Compute c-factor
    c = c_factor_from_metadata(metadata)

    # Extrapolate c-factor
    c = _extrapolate_c_factor(c)

    # Initialize progress bar
    pbar = tqdm(bands.items(), disable=quiet, leave=False)

    # Compute NBAR per band
    for band, resolution in pbar:
        pbar.set_description(f"Processing {band}")

        # Image file
        img_path = glob(
            os.path.join(
                path, "GRANULE", "*", "IMG_DATA", "*", f"*_{band}_{resolution}m.jp2"
            )
        )[0]

        # Rename filename to tif extension
        filename = os.path.split(img_path)[1].replace("jp2", "tif")

        # Open image and convert zeros to nan
        img = rioxarray.open_rasterio(img_path)
        img = img.where(lambda x: x > 0, other=np.nan)

        # Harmonize
        # This is a poor way of handling this as the xml file has
        # all the offsets and scale factors
        if harmonize:
            img = img - 1000

        # Interpolate c-factor of the band to the resolution of the image
        interpolated = c.sel(band=band).interp(
            y=img.y, x=img.x, method="linear", kwargs={"fill_value": "extrapolate"}
        )

        # Compute the NBAR
        img = img * interpolated

        if to_int:
            img = img.round().astype("int16")

        # Save the image
        img.rio.to_raster(os.path.join(nbar_output_path, filename), driver=driver)

    pbar.set_description("Done")

    # Show the path where the images were saved
    if not quiet:
        print(f"Saved to {nbar_output_path}")


def nbar_stac(
    ds: xr.Dataset, stac: str, collection: str, epsg: str, quiet: bool = False
) -> xr.Dataset:
    """Computes the Nadir BRDF Adjusted Reflectance (NBAR) for a :code:`xarray.DataArray`.

    For L2A STAC items taken from Microsoft Planetary Computer:
        If the processing baseline is greater than 04.00, the DN values
        are automatically shifted before computing NBAR.

    For L2A STAC items taken from AWS Element84
        The offset to subtract is given in the offset attributes
        of each data variable in the `ds`

    Parameters
    ----------
    ds : xr.Dataset {dims=(time, y, x)}
        Sentinel-2 L2A Dataset
    stac : str
        STAC Endpoint of the data array.
    collection : str
        Collection name of the data array.
    epsg : str
        EPSG code of the data array (e.g. "epsg:3115").
    quiet : bool, default = False
        Whether to show progress.

    Returns
    -------
    xr.Dataset {dims=(time, y, x)}
        NBAR - the BRDF corrected L2A dataset
    """
    # check whether the data was downloaded from Microsoft's planetary computer (PC)
    is_pc = "planetarycomputer" in stac
    is_aws = "aws.element84" in stac

    # Keep attributes xarray
    xr.set_options(keep_attrs=True)

    # 1. Get the xml url for each ID in the dataset
    xml_md = _get_xml_dict(ds, stac, collection)  # dict[str, dict[str, datetime | str]]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        session = _create_session(max_retries=3)

        # 2. Parallelise the download of the xml file (stored in `tmp_path`).
        #    The download is the slowest part of this entire function!!!
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for granule_id in xml_md:
                url = xml_md[granule_id]["xml"]
                time = xml_md[granule_id]["time"]
                futures[
                    executor.submit(
                        _fetch_xml, granule_id, url, time, tmp_path, session
                    )
                ] = granule_id

            for future in as_completed(futures):
                granule_id, time, xml_path = future.result()
                xml_md[granule_id]["local_xml"] = xml_path  # Path or None

        # 3. Extract the viewing and sun zenith and azimuthal angles from
        #    the xml file then compute the BRDF c-factors (Roy et al., 2017)
        corrected_datasets: list[xr.Dataset] = []
        ds_id_vals = ds.id.values
        band_vars = [k for k in ds.data_vars if k.lower() not in OTHER_DVARS]
        other_vars = [k for k in ds.data_vars if k.lower() in OTHER_DVARS]

        for granule_id in xml_md:
            xml_path = xml_md[granule_id]["local_xml"]
            time = xml_md[granule_id]["time"]

            # `c_id` is xr.Dataset dims=["y", "x"]
            c_id = c_factor_from_xml(xml_path, dst_crs=epsg, is_aws=is_aws)
            c_id.attrs = {"time": time, "id": granule_id}

            # interpolate `c_id` to match `y` and `x` of `ds`
            c_inter_id = c_id.interp(
                y=ds.y.values,
                x=ds.x.values,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            ).astype(np.float32)

            # time is not unique therefore use the ID for the matching
            time_match = [
                i for i in range(len(ds_id_vals)) if ds_id_vals[i] == granule_id
            ]
            n_mtime = len(time_match)
            if n_mtime != 1:
                msg = (
                    f"Expected a single match for {granule_id}, but found {n_mtime}\n:"
                    f"matching granule ids: {ds_id_vals[time_match]}"
                )
                raise ValueError(msg)

            # Select time slice and apply the scale and offsets
            ds_slice = ds.isel(time=time_match[0])

            # Apply the scale and offset to `ds_slice` and multiply the c-factors
            corrected_slice = (
                _apply_scale_offset(ds=ds_slice[band_vars], is_pc=is_pc)
                * c_inter_id[band_vars]  # some data variables maybe all np.nan's
            )
            corrected_slice = corrected_slice.where(
                np.isfinite(corrected_slice), other=np.nan
            )

            # Add back the SCL, AOT, WVP data variables to `corrected_slice`
            corrected_slice = corrected_slice.assign(
                variables={v: ds_slice[v] for v in other_vars}
            )
            corrected_datasets.append(corrected_slice)

    # Concatenate c-factor
    corrected_ds = xr.concat(corrected_datasets, dim="time")
    return corrected_ds


def nbar_stackstac(
    da: xr.DataArray, stac: str, collection: str, quiet: bool = False
) -> xr.DataArray:
    """Computes the Nadir BRDF Adjusted Reflectance (NBAR) for a :code:`xarray.DataArray`
    obtained via :code:`stackstac`.

    If the processing baseline is greater than 04.00, the DN values are automatically
    shifted before computing NBAR.

    Parameters
    ----------
    da : xarray.DataArray
        Data array obtained via :code:`stackstac` to use for the NBAR calculation.
    stac : str
        STAC Endpoint of the data array.
    collection : str
        Collection name of the data array.
    quiet : bool, default = False
        Whether to show progress.

    Returns
    -------
    xarray.DataArray
        NBAR data array.
    """
    # Get info from the stackstac data array
    epsg = da.attrs["crs"]

    # Compute NBAR
    da = nbar_stac(da, stac, collection, epsg, quiet)

    return da


def nbar_cubo(
    da: xr.DataArray | xr.Dataset, quiet: bool = False
) -> xr.DataArray | xr.Dataset:
    """Computes the Nadir BRDF Adjusted Reflectance (NBAR) for a :code:`xarray.DataArray`
    obtained via :code:`cubo`.

    If the processing baseline is greater than 04.00, the DN values are automatically
    shifted before computing NBAR.

    Parameters
    ----------
    da : xarray.DataArray
        Data array obtained via :code:`cubo` to use for the NBAR calculation.
    quiet : bool, default = False
        Whether to show progress.

    Returns
    -------
    xr.DataArray | xr.Dataset
        NBAR DataArray or Dataset
    """
    # Get info from the cubo data array
    stac = da.attrs["stac"]
    collection = da.attrs["collection"]
    epsg = da.attrs["epsg"]
    epsg = f"epsg:{epsg}"

    # Compute NBAR
    da = nbar_stac(da, stac, collection, epsg, quiet)

    return da
