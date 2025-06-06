"""
Functions used to acquire the detector footprints
"""

import requests
import geopandas as gpd
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from rasterio.crs import CRS


class Sentinel2DetFoo(object):
    def __init__(
        self, s2_tile_id: str, bands: list[str], dst_crs: CRS, max_workers: int = 4
    ):
        """
        Acquire Sentinel-2 detector footprints for a set of bands
        from the AWS-specific tile unique identifier.

        Parameters
        ----------
        s2_tile_id: str
            Sentinel-2 tile unique identifier, e.g. "S2B_50HMJ_20200605_1_L2A"
        bands : list[str]
            list of bands, must be ["B01", "B02", "B03", "B04", ... etc]
        dst_crs : CRS
            destination CRS used to reproject the detector footprint geometries
        max_workers : int
            Number of workers used for parallelisation
        """
        self.s2_tile_id = s2_tile_id
        self.bands = bands
        self.dst_crs = dst_crs
        self.max_workers = max_workers
        self.base_url = self.construct_base_url()
        return

    def construct_base_url(self) -> str:
        """
        Construct the urls for the gml files from the Sentinel-2
        unique tile identifier
        """
        parts = self.s2_tile_id.split("_")  # ["S2B" "50HMJ" 20200605" "1" "L2A"]

        mgrs = parts[1]  # "50HMJ"
        utm_zone = mgrs[:2]  # "50" (01 <= utm_zone <= 60)
        lat_band = mgrs[2]  # "H"
        grid_square = mgrs[3:]  # "MJ"

        date_str = parts[2]  # "20200605"
        year = date_str[:4]  # "2020"
        month = str(int(date_str[4:6]))  # 6
        day = str(int(date_str[6:8]))  # 5

        base_url = (
            "https://sentinel-s2-l2a.s3.amazonaws.com/tiles/"
            f"{utm_zone}/{lat_band}/{grid_square}/"
            f"{year}/{month}/{day}/0/qi/"
        )
        base_url += "MSK_DETFOO_{0}.gml"
        return base_url

    def fetch_gml(self, band: str) -> gpd.GeoDataFrame | None:
        url = self.base_url.format(band)
        gdf: gpd.GeoDataFrame | None = None

        if self.url_exists(url):
            gdf = gpd.read_file(url).to_crs(self.dst_crs)
            gdf["band"] = band

            if "gml_id" in gdf.columns:
                # extract the detector id from the "gml_id" column
                # e.g. "detector_footprint-B01-09-0", here the detector id = 9
                detector_id = gdf["gml_id"].str.split("-").str[2].astype(int)
                gdf["detector_id"] = detector_id
                gdf = gdf[["band", "detector_id", "geometry"]]
            else:
                raise ValueError("Unable to identify the detector id\n{gdf}")
        else:
            print(f"Failed to load {band} from {url}")

        return gdf

    def construct_gdf(self):
        """
        Acqure GML files and contruct the GeoDataFrame that
        contains the detector footprint for all bands
        """
        results: list[gpd.GeoDataFrame] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.fetch_gml, b): b for b in self.bands}

            for f in as_completed(futures):
                result = f.result()
                if result is not None:
                    results.append(result)

        if not results:
            raise ValueError("No valid GML files retrieved.")

        gdf_detfoo = gpd.GeoDataFrame(
            pd.concat(results, ignore_index=True), geometry="geometry"
        )
        gdf_detfoo.set_index(["band", "detector_id"], inplace=True)
        return gdf_detfoo

    @staticmethod
    def url_exists(url: str, timeout: int=3) -> bool:
        try:
            response = requests.head(url, timeout=timeout)
            return response.status_code < 400
        except requests.RequestException:
            return False
