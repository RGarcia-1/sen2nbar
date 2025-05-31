import os
import warnings

import numpy as np
import numpy.typing as npt
import requests
import xarray as xr
import xmltodict
from typing import Any


""" ASSIGN GLOBAL VARIABLES """
ANGLES = ["Zenith", "Azimuth"]

""" GLOBAL VARIABLES RELATED TO THE 23 x 23 ANGLE GRID """
GRID_SIZE = 23
GRID_RES = 5000  # metres


def _get_angle_values(
    values_dict: dict[str, dict[str, dict[str, list[str]]]], angle: str
) -> npt.NDArray[np.float32]:
    """Gets the angle values per detector in Sentinel-2 granule metadata.

    Parameters
    ----------
    values_dict : dict
        Dictionary of angle values in the metadata.
    angle : str
        Angle to retrieve. Either 'Zenith' oder 'Azimuth'.

    Returns
    -------
    numpy.ndarray
        Angle values per detector.
    """
    values = values_dict[angle]["Values_List"]["VALUES"]
    array = np.array([row.split(" ") for row in values]).astype(np.float32)
    return array


def _get_23x23_angle_grid(
    geo_info: dict[str, Any]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Get the x and y coordinates (in meters) of the 23 x 23 angular grid used
    in Sentinel-2 L2A tile metadata.

    This grid provides the coordinates of  the sun and view  angle values as
    reported in the  `Tile_Angles`  metadata section. The angular values are
    computed on a coarse 23 × 23 grid laid over the tile footprint with 5000
    metre spacing between nodes.

    https://forum.step.esa.int/t/view-and-sun-angles-grid-in-sentinel-2-data/43105/2
    According to ESA (via Florian - Sentinel-2 Technical Manager, OPT-MPC project):

        "The top left node of the grid has the same coordinates as the top
         left corner of the tile footprint (top left corner of the top left
         pixel of all the rasters at different spatial resolutions). The grid
         contains 23 by 23 nodes, each node being separated by 5000 m. The
         grid fully covers the tile footprint. It should be mentioned that
         the nodes from the last line and last column of the grid are 200 m
         away from the bottom and right tile edges, respectively."


    Parameters
    ----------
    geo_info : dict[str, Any]
        Dictionary from the Sentinel-2 L2A tile metadata containing
        'Tile_Geocoding' information.

    Returns
    -------
    x, y : npt.NDArray[np.float64]
        1D array of x and y coordinates (UTM meters) for the 23 angular grid columns.
    """
    tile_geocoding = geo_info["Tile_Geocoding"]
    res = float(tile_geocoding["Size"][0]["@resolution"])  # metres
    nrows = int(tile_geocoding["Size"][0]["NROWS"])
    ncols = int(tile_geocoding["Size"][0]["NCOLS"])
    x_ul = float(tile_geocoding["Geoposition"][0]["ULX"])  # upper left x
    y_ul = float(tile_geocoding["Geoposition"][0]["ULY"])  # upper left y
    # NOTE: `x_ul` and `y_ul` are the same for all resolutions (10, 20, 60 m)

    # Construct x and y center coordinate arrays for the 23x23 angle grid
    x_tmp = np.arange(GRID_SIZE) * GRID_RES
    y_tmp = np.arange(GRID_SIZE) * GRID_RES

    # At the moment x_tmp[22] and y_tmp[22] overshoots the footprint of
    # the array, that is, 
    #     x_tmp[22] > ncols * res, and;
    #     y_tmp[22] > nrows * res
    # Therefore, set the last grid node to 200 m before the tile edge.
    x_tmp[-1] = ncols * res - 200
    y_tmp[-1] = nrows * res - 200

    x = x_ul + x_tmp
    y = y_ul - y_tmp

    return x.astype(np.float64), y.astype(np.float64)


def angles_from_metadata(metadata: str) -> xr.DataArray:
    """Gets the angle values per band (and Sun) in Sentinel-2 granule metadata.

    The angle values are retrieved for the Sun and View modes as a
    :code:`xarray.DataArray` with a shape (band, angle, y, x).

    Parameters
    ----------
    metadata : str
        Path to the metadata file. An URL can also be used.

    Returns
    -------
    xarray.DataArray
        Angle values per band and Sun.
    """
    # Convert the xml into a dict
    if os.path.exists(metadata):
        data = xmltodict.parse(open(metadata, "r").read())
    else:
        data = xmltodict.parse(requests.get(metadata).content)

    # Extract the geocoding and angles, all the stuff we need is here
    Tile_Geocoding = data["n1:Level-2A_Tile_ID"]["n1:Geometric_Info"]["Tile_Geocoding"]
    tile_angles = data["n1:Level-2A_Tile_ID"]["n1:Geometric_Info"]["Tile_Angles"]

    x, y = _get_23x23_angle_grid(
        geo_info=data["n1:Level-2A_Tile_ID"]["n1:Geometric_Info"]
    )

    # Band names
    band_names = ["B" + f"0{x}"[-2:] for x in np.arange(1, 13)]
    band_names.insert(8, "B8A")

    # Create a dictionary to store the angles per band (and the Sun)
    bands_dict = dict()
    for key in ["Sun"] + band_names:
        bands_dict[key] = dict(Zenith=list(), Azimuth=list())

    # Each band has multiple detectors, so we have to go through all of them
    # and save them in a list to later do a nanmean
    for single_angle_detector in tile_angles["Viewing_Incidence_Angles_Grids"]:
        band_id = int(single_angle_detector["@bandId"])
        band_name = band_names[band_id]
        for angle in ANGLES:
            bands_dict[band_name][angle].append(
                _get_angle_values(single_angle_detector, angle)
            )

    # Do the same for the Sun, but there is just one, of course, duh
    for angle in ANGLES:
        bands_dict["Sun"][angle].append(
            _get_angle_values(tile_angles["Sun_Angles_Grid"], angle)
        )

    # Do the nanmean of the detectors angles per band
    filt_band_dict: dict[str, npt.NDArray[np.floating]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for band_name in bands_dict:
            for angle in ANGLES:
                bands_dict[band_name][angle] = np.nanmean(
                    np.array(bands_dict[band_name][angle]), axis=0
                )
            filt_band_dict[band_name] = np.array(
                [bands_dict[band_name][angle] for angle in ANGLES]
            )  # dims = [num_angles, len(x), len(y)

    from pprint import pprint
    pprint(filt_band_dict)
    exit()
    # Create the array
    try:
        da = xr.DataArray(
            list(bands_dict.values()),
            dims=["band", "angle", "y", "x"],
            coords=dict(band=list(bands_dict.keys()), angle=ANGLES, x=x, y=y),
        )
    except ValueError:
        raise ValueError("Not all bands include angles values.")

    # Add attributes
    da.attrs["epsg"] = Tile_Geocoding["HORIZONTAL_CS_CODE"]

    return da


def get_processing_baseline(metadata: str) -> float:
    """Gets the processing baseline in Sentinel-2 user metadata.

    The processing baseline is retrieved as a float.

    Parameters
    ----------
    metadata : str
        Path to the metadata file. An URL can also be used.

    Returns
    -------
    float
        Processing baseline.
    """
    # Convert the xml into a dict
    if os.path.exists(metadata):
        data = xmltodict.parse(open(metadata, "r").read())
    else:
        data = xmltodict.parse(requests.get(metadata).content)

    # Get the processing baseline
    PROCESSING_BASELINE = data["n1:Level-2A_User_Product"]["n1:General_Info"][
        "Product_Info"
    ]["PROCESSING_BASELINE"]

    return float(PROCESSING_BASELINE)
