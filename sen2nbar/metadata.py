import os
import warnings

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt  # noqa: F401
import requests
import xarray as xr
import xmltodict
import defusedxml.ElementTree as ET
from affine import Affine
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from scipy.interpolate import RegularGridInterpolator, RBFInterpolator
from shapely.plotting import plot_polygon
from xml.etree.ElementTree import Element

from .det import Sentinel2DetFoo


""" ASSIGN GLOBAL VARIABLES """
BNAME_TO_ID = {  # verified in <Spectral_Information_List> of MTD_MSIL1C.xml
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


def spatial_info_at_res(
    root: Element, namespaces: dict[str, str], res: int = 10
) -> dict[str, str]:
    """
    Get spatial information (shape, Affine transformation matrix, crs) of
    sentinel-2 image for a given resolution. This information could be used
    to reproject the angle grid to the desired resolution
    """
    geo_info = root.find(".//n1:Geometric_Info", namespaces=namespaces)
    geocoding = geo_info.find("Tile_Geocoding")

    size = geocoding.find(f'Size[@resolution="{res}"]')
    geopos = geocoding.find(f'Geoposition[@resolution="{res}"]')
    crs = CRS.from_user_input(geocoding.find("HORIZONTAL_CS_CODE").text)

    nrows = int(size.find("NROWS").text)
    ncols = int(size.find("NCOLS").text)

    # Get the top-left corner of the top-left pixel of the grid:
    ulx = float(geopos.find("ULX").text)
    uly = float(geopos.find("ULY").text)

    reproj_kw = {
        "shape": (nrows, ncols),
        "crs": crs,
        "transform": Affine.translation(ulx, uly) * Affine.scale(res, -res),
    }
    return reproj_kw


def get_xy_angle_grid(
    root: Element, namespaces: dict[str, str], res: int = 10
) -> tuple[FLOAT_ARRAY, FLOAT_ARRAY, CRS]:
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

    Notes:
    ------
    -> There is no need to shift these coordinates to match the centre
       of the pixel because GDAL (and hence rasterio) defines the origin
       as the top-left corner of the top-left pixel:

       https://gdal.org/en/stable/user/raster_data_model.html#affine-geotransform

      "Affine GeoTransform:
       Note that the pixel/line coordinates in the above are from (0.0,0.0) at the top
       left corner of the top left pixel to (width_in_pixels, height_in_pixels) at the
       bottom right corner of the bottom right pixel."
    -> The output is raster with an irregular grid (owing to the last row and column).
       Use scipy.interpolate.RegularGridInterpolator to resample, as this function,
       the retangular grid can have even or uneven spacing.

    Returns
    -------
    x, y : npt.NDArray[np.float64]
        1D array of x and y coordinates (UTM meters) for the 23 angular grid
        columns. These values represent the centre of each pixel, with the
        last row and last column existing out-of-bounds.
    crs : CRS
        CRS of the x and y coordinates
    """
    geo_info = root.find(".//n1:Geometric_Info", namespaces=namespaces)
    geocoding = geo_info.find("Tile_Geocoding")

    size = geocoding.find(f'Size[@resolution="{res}"]')
    geopos = geocoding.find(f'Geoposition[@resolution="{res}"]')
    crs = CRS.from_user_input(geocoding.find("HORIZONTAL_CS_CODE").text)

    nrows = int(size.find("NROWS").text)
    ncols = int(size.find("NCOLS").text)

    # Get the top-left corner of the top-left pixel of the grid:
    ulx = float(geopos.find("ULX").text)
    uly = float(geopos.find("ULY").text)

    # Construct x and y center coordinate arrays for the 23x23 angle grid
    x_tmp = np.arange(GRID_SIZE) * GRID_RES
    y_tmp = np.arange(GRID_SIZE) * GRID_RES

    # At the moment the top-left corner of x_tmp[22] and y_tmp[22]
    # overshoots the footprint of the array, that is,
    #     x_tmp[22] > ncols * res, and;
    #     y_tmp[22] > nrows * res
    # Therefore, set the top-left corner of the last grid node to
    # 200 m before the tile edge (see above)
    x_tmp[-1] = ncols * res - 200
    y_tmp[-1] = nrows * res + 200

    x = ulx + x_tmp
    y = uly - y_tmp

    return x.astype("float64"), y.astype("float64"), crs


def get_values(elt, item_tag: str) -> FLOAT_ARRAY:
    elts = elt.findall(f"{item_tag}/Values_List/VALUES")
    values = [[float(v) for v in row.text.split()] for row in elts]
    return np.array(values, dtype="float32")


class Sentinel2ViewAngleParser:
    def __init__(
        self,
        xml_path: str | Path,
        band_names: list[str],
        detfoo_gdf: gpd.GeoDataFrame,
        dst_res: float = 10.0,
    ):
        """
        Sentinel2ViewAngleParser

        Parameters
        ----------
        xml_path : str | Path
            Path to local xml file or the URL
        band_names : list[str]
            Band names
        detfoo_gdf : gpd.GeoDataFrame
            GeoDataFrame containing the detector footprint per band
        dst_res : float
            Destination resolution
        """
        self.band_names = band_names
        self.band_ids = [BNAME_TO_ID[b] for b in self.band_names]
        self.xml_content = self._get_xml_contents(xml_path)
        self.root = ET.fromstring(self.xml_content)
        self.detfoo_gdf = detfoo_gdf
        self.dst_res = dst_res

        # Get the namespace from the root
        self.view_key = "Viewing_Incidence_Angles_Grids[@bandId='{0}']"
        self.ns: dict[str, str] = {"n1": self.root.tag.split("}")[0].strip("{")}
        self.tile_angles = self.root.find(".//Tile_Angles", namespaces=self.ns)

        # Get the angle grid raster information
        x, y, crs = get_xy_angle_grid(self.root, self.ns, int(self.dst_res))
        self.grid_x = x
        self.grid_y = y
        self.grid_crs = crs

    def _get_xml_contents(self, source: str | Path) -> bytes:
        if "http" in str(source):
            response = requests.get(source)
            response.raise_for_status()
            return response.content
        else:
            with open(source, "rb") as f:
                return f.read()

    def _extrapolate(self, course_grid: FLOAT_ARRAY) -> FLOAT_ARRAY:
        """
        Extrapolate over NaN value in the 23 x 23 `course_grid`
        """
        valid_ix = np.isfinite(course_grid)
        if not np.any(valid_ix):
            return course_grid

        y_idx, x_idx = np.indices(course_grid.shape)
        interp = RBFInterpolator(
            y=np.column_stack((y_idx[valid_ix], x_idx[valid_ix])),
            d=course_grid[valid_ix],
        )

        # Extrapolate
        coords = np.column_stack((y_idx[~valid_ix], x_idx[~valid_ix]))

        filled_grid = course_grid.copy()
        filled_grid[~valid_ix] = interp(coords)
        return filled_grid

    def _upscale_grid(
        self, course_grid: FLOAT_ARRAY
    ) -> tuple[FLOAT_ARRAY, dict[str, Any]]:
        """
        Upscale the rectilinear angle grid to a regular grid with spacing
        defined by `self.dst_res`
        """
        spatial_md = spatial_info_at_res(self.root, self.ns, int(self.dst_res))

        # We first need to extrapolate NaN values
        filled_grid = self._extrapolate(course_grid)

        # Define the interpolator - NOTE: the points must be strictly
        # ascending or descending. Given that `grid_x` is ascending
        # while `grid_y` is descending, the data along axis=0 (y)
        # will need to be flipped.
        flipped_grid_y = np.flip(self.grid_y, axis=0)  # or use np.sort()
        interp = RegularGridInterpolator(
            points=(flipped_grid_y, self.grid_x),
            values=np.flip(filled_grid, axis=0),
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        # Define regular output grid
        rows = np.arange(spatial_md["shape"][0])
        cols = np.arange(spatial_md["shape"][1])

        x0, delta_x = spatial_md["transform"].c, spatial_md["transform"].a
        y0, delta_y = spatial_md["transform"].f, spatial_md["transform"].e

        x_out = x0 + cols * delta_x
        y_out = np.flip(y0 + rows * delta_y, axis=0)  # flip ascending order
        xx_out, yy_out = np.meshgrid(x_out, y_out)

        # Interpolate
        coords = np.column_stack([yy_out.ravel(), xx_out.ravel()])
        resampled = np.flip(interp(coords).reshape(yy_out.shape), axis=0)
        return resampled, spatial_md

    def visualise_angle_detfoo(
        self, angle: FLOAT_ARRAY, detfoo: gpd.GeoDataFrame
    ) -> None:
        left = self.grid_x.min()
        right = self.grid_x.max()
        top = self.grid_y.max()
        bottom = self.grid_y.min()

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(
            angle, interpolation="None", cmap="jet", extent=[left, right, bottom, top]
        )
        plot_polygon(
            detfoo.geometry.values[0], ax=ax, add_points=False, color="gray", alpha=0.7
        )
        plt.show()
        exit()
        return

    def _process_band(self, band_name: str) -> tuple[str, FLOAT_ARRAY, FLOAT_ARRAY]:
        band_id = BNAME_TO_ID[band_name]
        grids = self.tile_angles.findall(
            self.view_key.format(band_id), namespaces=self.ns
        )
        print(f"{band_id=}, {band_name=}")
        for g in grids:
            # get the detector ID
            det_id = g.attrib.get("detectorId", None)
            if det_id is None:
                msg = f"Unable to extract the detector ID: {g.attrib}"
                raise ValueError(msg)

            det_azi, meta = self._upscale_grid(course_grid=get_values(g, AZI_TAG))
            detfoo = self.detfoo_gdf.loc[
                (self.detfoo_gdf["band"] == band_name)
                & (self.detfoo_gdf["detector_id"] == int(det_id))
            ]
            geom_mask = geometry_mask(
                geometries=detfoo.geometry.values,
                out_shape=det_azi.shape,
                transform=meta["transform"],
                all_touched=True,
                invert=True,
            )
            det_azi[~geom_mask] = np.nan
            self.visualise_angle_detfoo(det_azi, detfoo)
            exit()
        exit()

        shape = [GRID_SIZE, GRID_SIZE]

        if not grids:
            zen = np.full(shape, np.nan, dtype="float32")
            azi = np.full(shape, np.nan, dtype="float32")
        else:
            zen = np.nanmean(np.stack([get_values(g, ZEN_TAG) for g in grids]), axis=0)
            azi = np.nanmean(np.stack([get_values(g, AZI_TAG) for g in grids]), axis=0)

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
        zen_data: dict[str, tuple(list[str], FLOAT_ARRAY)] = {}
        azi_data: dict[str, tuple(list[str], FLOAT_ARRAY)] = {}
        dims = ["y", "x"]
        for band_name in self.band_names:
            print(f"Here: {band_name}")
            _, zenith, azimuth = self._process_band(band_name)
            exit()
            zen_data[band_name] = (dims, zenith)
            azi_data[band_name] = (dims, azimuth)

        # Now extract the solar angles
        sun_grid = self.tile_angles.find("Sun_Angles_Grid")
        zen_data["sun"] = (dims, get_values(sun_grid, ZEN_TAG))
        azi_data["sun"] = (dims, get_values(sun_grid, AZI_TAG))

        coords: dict[str, Any] = {
            "band": list(zen_data.keys()),
            "y": self.grid_y,
            "x": self.grid_x,
        }
        attrs: dict[str, str] = {"crs": self.grid_crs}

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
        zen_data["sun"] = (dims, get_values(sun_grid, ZEN_TAG))
        azi_data["sun"] = (dims, get_values(sun_grid, AZI_TAG))

        coords: dict[str, Any] = {
            "band": list(zen_data.keys()),
            "y": self.grid_y,
            "x": self.grid_x,
        }
        attrs: dict[str, str] = {"crs": self.grid_crs}

        zen_ds = xr.Dataset(data_vars=zen_data, coords=coords, attrs=attrs)
        azi_ds = xr.Dataset(data_vars=azi_data, coords=coords, attrs=attrs)

        return zen_ds, azi_ds


def angles_from_metadata(
    tile_id: str, metadata_xml: str, crs: CRS, multiproc: bool = True
) -> tuple[xr.Dataset, xr.Dataset]:
    """Gets the angle values per band (and Sun) in Sentinel-2 granule metadata.

    The angle values are retrieved for the Sun and View modes as a
    `xarray.DataArray` with a shape (band, angle, y, x).

    Parameters
    ----------
    tile_id : str
        Unique Sentinel 2 tile identifier (e.g. "S2B_50HMJ_20200605_1_L2A")
    metadata_xml : str
        Path to the metadata xml file. An URL can also be used.
    crs : CRS
        Coordinate Reference System used to reproject viewing and solar
        angles.
    multiproc : bool
        Whether to use parallel processing.

    Returns
    -------
    zenith_ds, azimuth_ds : xr.Dataset (dims=["bands", "y", "x"])
        zenith and azimuth angles for the 23 x 23 grid for all bands
        including the solar angles
    """

    band_names = ["B" + f"0{x}"[-2:] for x in np.arange(1, 13)]
    band_names.insert(8, "B8A")

    ofile = Path(
        f"/home/rgarcia/Documents/tasks/misc/cp_get_sen2/{tile_id}_detfoo.gpkg"
    )
    if not ofile.exists():
        sdf = Sentinel2DetFoo(tile_id, bands=band_names, dst_crs=crs, max_workers=12)
        detfoo_gdf = sdf.construct_gdf()
        detfoo_gdf.to_file(ofile, driver="GPKG")
        print(f"created: {ofile}")
    else:
        detfoo_gdf = gpd.read_file(ofile)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        s2_xml = Sentinel2ViewAngleParser(
            xml_path=metadata_xml, band_names=band_names, detfoo_gdf=detfoo_gdf
        )

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
