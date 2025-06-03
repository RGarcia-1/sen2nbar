import os
import warnings

import numpy as np
import numpy.typing as npt
import requests
import xarray as xr
import xmltodict
import defusedxml.ElementTree as ET
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element
from concurrent.futures import ThreadPoolExecutor, as_completed


""" ASSIGN GLOBAL VARIABLES """
BAND_NAME_TO_ID = {  # verified in <Spectral_Information_List> of MTD_MSIL1C.xml
    "B01": "0",
    "B02": "1",
    "B03": "2",
    "B04": "3",
    "B05": "4",
    "B06": "5",
    "B07": "6",
    "B08": "7",
    "B8A": "8",
    "B09": "9",
    "B10": "10",
    "B11": "11",
    "B12": "12",
}

""" GLOBAL VARIABLES RELATED TO THE 23 x 23 ANGLE GRID """
GRID_SIZE = 23
GRID_RES = 5000  # metres
ZEN_TAG = "Zenith"
AZI_TAG = "Azimuth"

FLOAT_ARRAY = npt.NDArray[np.floating]


def _get_xy_angle_grid(
    root: Element, namespaces: dict[str, str]
) -> tuple[FLOAT_ARRAY, FLOAT_ARRAY, str]:
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

    Returns
    -------
    x, y : npt.NDArray[np.float64]
        1D array of x and y coordinates (UTM meters) for the 23 angular grid columns.
    crs : str
        EPSG code of the x and y coordinates
    """
    res = 10
    geo_info = root.find(".//n1:Geometric_Info", namespaces=namespaces)
    geocoding = geo_info.find("Tile_Geocoding")

    size = geocoding.find(f'Size[@resolution="{res}"]')
    geopos = geocoding.find(f'Geoposition[@resolution="{res}"]')

    nrows = int(size.find("NROWS").text)
    ncols = int(size.find("NCOLS").text)
    ulx = float(geopos.find("ULX").text)
    uly = float(geopos.find("ULY").text)
    crs = geocoding.find("HORIZONTAL_CS_CODE").text

    # Construct x and y center coordinate arrays for the 23x23 angle grid
    x_tmp = np.arange(GRID_SIZE) * GRID_RES
    y_tmp = np.arange(GRID_SIZE) * GRID_RES

    # At the moment x_tmp[22] and y_tmp[22] overshoots the footprint of
    # the array, that is,
    #     x_tmp[22] > ncols * res, and;
    #     y_tmp[22] > nrows * res
    # Therefore, set the last grid node to 200 m before the tile edge.
    x_tmp[-1] = ncols * res - 200
    y_tmp[-1] = nrows * res + 200

    x = ulx + x_tmp
    y = uly - y_tmp

    return x.astype("float64"), y.astype("float64"), crs


def _get_values(elt, item_tag: str) -> FLOAT_ARRAY:
    elts = elt.findall(f"{item_tag}/Values_List/VALUES")
    values = [[float(v) for v in row.text.split()] for row in elts]
    return np.array(values, dtype="float32")


class Sentinel2ViewAngleParser:
    def __init__(self, xml_path: str | Path, band_names: list[str]):
        self.band_names = band_names
        self.band_ids = [BAND_NAME_TO_ID[b] for b in self.band_names]
        self.xml_content = self._get_xml_contents(xml_path)
        self.root = ET.fromstring(self.xml_content)

        # Get the namespace from the root
        self.view_key = "Viewing_Incidence_Angles_Grids[@bandId='{0}']"
        self.ns: dict[str, str] = {"n1": self.root.tag.split("}")[0].strip("{")}
        self.tile_angles = self.root.find(".//Tile_Angles", namespaces=self.ns)

    def _get_xml_contents(self, source: str | Path):
        if "http" in str(source):
            response = requests.get(source)
            response.raise_for_status()
            return response.content
        else:
            with open(source, "rb") as f:
                return f.read()

    def _process_band(self, band_name: str) -> tuple[str, FLOAT_ARRAY, FLOAT_ARRAY]:
        band_id = BAND_NAME_TO_ID[band_name]
        grids = self.tile_angles.findall(
            self.view_key.format(band_id), namespaces=self.ns
        )

        shape = [GRID_SIZE, GRID_SIZE]

        if not grids:
            zen = np.full(shape, np.nan, dtype="float32")
            azi = np.full(shape, np.nan, dtype="float32")
        else:
            zen = np.nanmean(np.stack([_get_values(g, ZEN_TAG) for g in grids]), axis=0)
            azi = np.nanmean(np.stack([_get_values(g, AZI_TAG) for g in grids]), axis=0)

        return band_name, zen, azi

    def get_angles_ds_single(self) -> tuple[xr.Dataset, xr.Dataset]:
        """
        Get view and azimuth angle values for selected bands

        Returns:
        --------
        zen_ds, azi_ds : xr.Dataset (dims=["bands", "y", "x"])
            zenith and azimuth angles for the 23 x 23 grid for all bands
            including the solar angles
        """
        x, y, crs = _get_xy_angle_grid(self.root, self.ns)

        zen_data: dict[str, tuple(list[str], FLOAT_ARRAY)] = {}
        azi_data: dict[str, tuple(list[str], FLOAT_ARRAY)] = {}
        dims = ["y", "x"]
        for band_name in self.band_names:
            _, zenith, azimuth = self._process_band(band_name)
            zen_data[band_name] = (dims, zenith)
            azi_data[band_name] = (dims, azimuth)

        # Now extract the solar angles
        sun_grid = self.tile_angles.find("Sun_Angles_Grid")
        zen_data["sun"] = (dims, _get_values(sun_grid, ZEN_TAG))
        azi_data["sun"] = (dims, _get_values(sun_grid, AZI_TAG))

        coords: dict[str, Any] = {
            "band": list(zen_data.keys()),
            "y": y,
            "x": x,
        }
        attrs: dict[str, str] = {"crs": crs}

        zen_ds = xr.Dataset(data_vars=zen_data, coords=coords, attrs=attrs)
        azi_ds = xr.Dataset(data_vars=azi_data, coords=coords, attrs=attrs)

        return zen_ds, azi_ds

    def get_angles_ds_multi(self) -> tuple[xr.Dataset, xr.Dataset]:
        """
        Get view and azimuth angle values for selected bands (Multiprocessing)

        Returns:
        --------
        zen_ds, azi_ds : xr.Dataset (dims=["bands", "y", "x"])
            zenith and azimuth angles for the 23 x 23 grid for all bands
            including the solar angles
        """
        x, y, crs = _get_xy_angle_grid(self.root, self.ns)

        zen_data: dict[str, tuple(list[str], FLOAT_ARRAY)] = {}
        azi_data: dict[str, tuple(list[str], FLOAT_ARRAY)] = {}
        dims = ["y", "x"]

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._process_band, band_name): band_name
                for band_name in self.band_names
            }
            for future in as_completed(futures):
                band, zen, azi = future.result()
                zen_data[band] = (dims, zen)
                azi_data[band] = (dims, azi)

        # Now extract the solar angles
        tile_angles = self.root.find(".//Tile_Angles", namespaces=self.ns)
        sun_grid = tile_angles.find("Sun_Angles_Grid")
        zen_data["sun"] = (dims, _get_values(sun_grid, ZEN_TAG))
        azi_data["sun"] = (dims, _get_values(sun_grid, AZI_TAG))

        coords: dict[str, Any] = {
            "band": list(zen_data.keys()),
            "y": y,
            "x": x,
        }
        attrs: dict[str, str] = {"crs": crs}

        zen_ds = xr.Dataset(data_vars=zen_data, coords=coords, attrs=attrs)
        azi_ds = xr.Dataset(data_vars=azi_data, coords=coords, attrs=attrs)

        return zen_ds, azi_ds


def angles_from_metadata(
    metadata: str, multiproc: bool = True
) -> tuple[xr.Dataset, xr.Dataset]:
    """Gets the angle values per band (and Sun) in Sentinel-2 granule metadata.

    The angle values are retrieved for the Sun and View modes as a
    :code:`xarray.DataArray` with a shape (band, angle, y, x).

    Parameters
    ----------
    metadata : str
        Path to the metadata file. An URL can also be used.

    Returns
    -------
    zenith_ds, azimuth_ds : xr.Dataset (dims=["bands", "y", "x"])
        zenith and azimuth angles for the 23 x 23 grid for all bands
        including the solar angles
    """

    band_names = ["B" + f"0{x}"[-2:] for x in np.arange(1, 13)]
    band_names.insert(8, "B8A")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        s2_xml = Sentinel2ViewAngleParser(xml_path=metadata, band_names=band_names)

        if multiproc:
            zenith_ds, azimuth_ds = s2_xml.get_angles_ds_multi()
        else:
            zenith_ds, azimuth_ds = s2_xml.get_angles_ds_single()

    return zenith_ds, azimuth_ds


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
