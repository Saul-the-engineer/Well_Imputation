import utils
import os
import logging
import math
import fiona
import numpy as np
import pandas as pd
import netCDF4 as nc
import datetime as dt
import shapely.geometry
import gstools as gs
import matplotlib.pyplot as plt

from netCDF4 import Variable
from typing import Tuple, Union, List
from shapely.geometry import MultiPolygon, shape, Point
from shapely.ops import unary_union

logging.basicConfig(
    filename="storage_change.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
)


def kriging_interpolation(
    data_pickle_path: str,
    shape_file_path: str,
    n_x_cells: int = 100,
    monthly_time_step: int = 1,
    netcdf_filename: str = "interpolation.nc",
    directory: str = "./",
    influence_distance: float = 0.125,
):
    # TODO Add ability to specify resolution based on number of cells along the y-axis
    data = utils.load_pickle(data_pickle_path)
    assert isinstance(data, dict), "Data must be a dictionary."
    assert "Data" in data.keys(), "Data must contain a 'Data' key."
    assert "Location" in data.keys(), "Data must contain a 'Location' key."

    # parse data from dictionary
    data_measurements = utils.filter_dataframe_by_monthly_step(
        data["Data"].dropna(axis=1), monthly_time_step
    )
    # parse data from dictionary
    x_coordinates = data["Location"]["Longitude"]
    y_coordinates = data["Location"]["Latitude"]
    timestamps = data_measurements.index

    polygon = utils.load_shapefile(path=shape_file_path)
    grid_longitude, grid_latitude, mask_array = create_spatial_grid(
        polygon=polygon, n_x_cells=n_x_cells
    )
    file_nc, raster_data = create_netcdf_file(
        longitude=grid_longitude,
        latitude=grid_latitude,
        timestamp=timestamps,
        filename=netcdf_filename,
        data_root=directory,
    )

    create_interpolated_surfaces(
        timeseries_data=data_measurements,
        longitude=x_coordinates,
        latitude=y_coordinates,
        grid_longitude=grid_longitude,
        grid_latitude=grid_latitude,
        file_nc=file_nc,
        raster_data=raster_data,
        mask_array=mask_array,
        influence_distance=influence_distance,
    )
    return


def create_spatial_grid(
    polygon: fiona.Collection, n_x_cells: int = 100
) -> Tuple[list, list]:
    """
    Create a grid of points within a polygon.

    Returns a grid of points within a polygon. The grid is created based on the
    bounding box of the polygon. The resolution of the grid is determined by the
    number of cells along the x-axis.

    :param polygon: The polygon to create a grid within.
    :type polygon: fiona.Collection
    :param n_x_cells: The number of cells along the x-axis.
    :type n_x_cells: int, optional
    :return: The grid of points within the polygon.
    :rtype: Tuple[list, list]
    """
    assert isinstance(polygon, fiona.Collection), "Polygon must be a fiona collection."
    assert isinstance(n_x_cells, int), "n_x_cells must be an integer."
    assert n_x_cells is not None, "n_x_cells must be specified."
    assert n_x_cells > 0, "n_x_cells must be greater than 0."

    # Calculate the bounding box of the polygon
    polygon_boundary = polygon.bounds
    south_east_lon, south_east_lat, north_west_lon, north_west_lat = polygon_boundary

    # Calculate the longitude and latitude range
    longitude_range = abs(north_west_lon - south_east_lon)
    latitude_range = abs(north_west_lat - south_east_lat)
    # TODO: Add ability to specify resolution without reference to number of cells in shape
    # TODO: Add ability to specify resolution based on number of cells along the y-axis
    resolution = float(longitude_range / n_x_cells)

    logging.info(f"Longitude range is: {np.round(longitude_range,3)}.")
    logging.info(f"Latitude range is: {np.round(latitude_range,3)}.")
    logging.info(f"Grid Resolution is {np.round(resolution,3)}.")

    # Create grid arrays using NumPy
    grid_latitude = np.arange(north_west_lat, south_east_lat, -resolution)
    grid_longitude = np.arange(south_east_lon, north_west_lon, resolution)

    # Create the mask_array with NumPy
    mask_array = np.ones((len(grid_latitude), len(grid_longitude)), dtype=float)
    poloygon_vector_boundary = _extract_vector_boundary(polygon)

    # Loop through every point to see if point is in shape
    for i, lat in enumerate(grid_latitude):
        for j, lon in enumerate(grid_longitude):
            point = Point(lon, lat)
            if not poloygon_vector_boundary.contains(point):
                mask_array[i, j] = 0

    # Mask the array to be used in kriging
    mask_array = np.where(mask_array == 0, np.nan, 1)

    return grid_longitude, grid_latitude, mask_array


def _extract_vector_boundary(polygons):
    shapes = []
    for polygon in polygons:
        if polygon["geometry"]["type"] == "Polygon":
            shapes.append(shape(polygon["geometry"]))
        elif polygon["geometry"]["type"] == "MultiPolygon":
            multi_poly = shape(polygon["geometry"])
            for (
                sub_polygon
            ) in (
                multi_poly.geoms
            ):  # Iterate over individual polygons within MultiPolygon
                shapes.append(sub_polygon)
    boundary = unary_union(shapes)
    return boundary


def create_netcdf_file(
    longitude: List[float],
    latitude: List[float],
    timestamp: pd.DatetimeIndex,
    filename: str,
    data_root: str,
) -> Union[nc.Dataset, nc.Variable]:
    CALENDAR = "standard"
    UNITS = "days since 0001-01-01 00:00:00"
    try:
        file = nc.Dataset(f"{data_root}/{filename}", "w", format="NETCDF4")

        lon_len = len(longitude)
        lat_len = len(latitude)

        time_dim = file.createDimension("time", None)
        lat_dim = file.createDimension("lat", lat_len)
        lon_dim = file.createDimension("lon", lon_len)

        time_var = file.createVariable("time", np.float64, ("time"))
        lat_var = file.createVariable("lat", np.float64, ("lat"))
        lon_var = file.createVariable("lon", np.float64, ("lon"))
        tsvalue_var = file.createVariable(
            "tsvalue", np.float64, ("time", "lat", "lon"), fill_value=-9999
        )

        lat_var[:] = latitude
        lon_var[:] = longitude

        lat_var.long_name = "Latitude"
        lat_var.units = "degrees_north"
        lat_var.axis = "Y"

        lon_var.long_name = "Longitude"
        lon_var.units = "degrees_east"
        lon_var.axis = "X"

        timestamp_list = list(timestamp.to_pydatetime())
        time_var[:] = nc.date2num(dates=timestamp_list, units=UNITS, calendar=CALENDAR)
        time_var.axis = "T"
        time_var.units = UNITS

        return file, tsvalue_var
    except Exception as e:
        # Handle any exceptions gracefully
        print(f"Error creating NetCDF file: {str(e)}")
        return None, None


def create_interpolated_surfaces(
    timeseries_data: pd.DataFrame,
    longitude: pd.Series,
    latitude: pd.Series,
    grid_longitude: np.ndarray,
    grid_latitude: np.ndarray,
    file_nc: nc.Dataset,
    raster_data: nc.Variable,
    mask_array: np.ndarray = None,
    influence_distance: float = 0.125,
):
    """
    Interpolate Time Series Data and Create a NetCDF File

    This function performs spatial interpolation on time series data, fits variogram models, and creates a NetCDF file to store the interpolated results.

    :param timeseries_data: A DataFrame containing time series data for interpolation.
    :type timeseries_data: pd.DataFrame
    :param longitude: A Series of longitude values indexed by well name.
    :type longitude: pd.Series
    :param latitude: A Series of latitude values indexed by well name.
    :type latitude: pd.Series
    :param grid_longitude: An array of longitude values for the grid.
    :type grid_longitude: np.ndarray
    :param grid_latitude: An array of latitude values for the grid.
    :type grid_latitude: np.ndarray
    :param file_nc: A NetCDF dataset for storing the interpolated data.
    :type file_nc: nc.Dataset
    :param raster_data: A NetCDF variable for storing the interpolated raster data.
    :type raster_data: nc.Variable
    :param mask_array: A binary mask array to apply to the kriging result (default: None).
    :type mask_array: np.ndarray, optional
    :return: The interpolated surface data.
    :rtype: gs.krige.Ordinary
    """
    for i, date in enumerate(timeseries_data.index):
        logging.info(f"Processing date: {date}")
        timeseries_step = timeseries_data.loc[date].dropna(axis=0)
        values = timeseries_step.values
        point_coordinates = (
            longitude.loc[timeseries_step.index],
            latitude.loc[timeseries_step.index],
        )

        variogram = fit_model_variogram(
            point_coordinates=point_coordinates,
            values=values,
            influence_distance=influence_distance,
        )
        surface = interpolate_surface(
            variogram=variogram,
            point_coords=point_coordinates,
            timeseries_values=values,
            grid_x=grid_longitude,
            grid_y=grid_latitude,
            mask_array=mask_array,
        )
        raster_data[i, :, :] = surface.field.T
    file_nc.sync()
    file_nc.close()
    logging.info("NetCDF file created.")
    return surface


def fit_model_variogram(
    point_coordinates: Tuple[np.ndarray, np.ndarray],
    values: np.ndarray,
    influence_distance: float = 0.125,
) -> gs.Stable:
    """
    Fit a variogram model to spatial data.

    Calculates the experimental variogram and fits a synthetic variogram model to it.

    :param point_coordinates: Array of longitude and latitude coordinates.
    :type longitude_array: Tuple[np.ndarray, np.ndarray]
    :param values: Array of spatial values.
    :type values: np.ndarray
    :param influence: Influence of the model.
    :type influence: float, optional
    :param bin_num: Number of bins for the variogram.
    :type bin_num: int, optional
    :param plot: Whether to plot the variogram.
    :type plot: bool, optional
    :return: The experimental and synthetic variograms.
    :rtype: gs.Stable
    """
    longitude_array, latitude_array = point_coordinates
    x_delta = np.max(longitude_array) - np.min(longitude_array)
    y_delta = np.max(latitude_array) - np.min(latitude_array)
    # determine the maximum distance between points in our shape.
    max_dist = np.sqrt(x_delta**2 + y_delta**2)
    # scale the variogram range to a percentage of the maximum distance between points.
    variogram_range = max_dist * influence_distance

    sill = np.var(values)

    # fit the model to a synthetic variogram. The nugget is set to 0.0 because we don't expect
    # any error in the data. The anisotropy is set to 1.0 because we don't expect any
    # anisotropy in the data.
    variogram = gs.Stable(
        dim=2, var=sill, len_scale=variogram_range, nugget=0.0, anis=1.0
    )

    return variogram


def interpolate_surface(
    variogram: gs.Stable,
    point_coords: Tuple[np.ndarray],
    timeseries_values: np.ndarray,
    grid_x: List[float],
    grid_y: List[float],
    mask_array: np.ndarray = None,
) -> gs.krige.Ordinary:
    """
    Kriging timeseries data to generate a surface map.

    :param variogram: The variogram model used for kriging.
    :type variogram: gs.Stable
    :param point_coords: Coordinates of the well data.
    :type point_coords: Tuple[np.ndarray]
    :param timeseries_values: Values of the timeseries data.
    :type timeseries_values: np.ndarray
    :param grid_x: X-coordinates of the grid.
    :type grid_x: List[float]
    :param grid_y: Y-coordinates of the grid.
    :type grid_y: List[float]
    :param mask_array: Binary mask array to apply to the kriging result to remove values outside of the shape.
    :type mask_array: np.ndarray
    :return: Kriging result.
    :rtype: gs.krige.Ordinary
    """

    krig_map = gs.krige.Ordinary(
        model=variogram, cond_pos=point_coords, cond_val=timeseries_values
    )
    krig_map.structured([grid_x, grid_y])

    if mask_array is not None:
        krig_map.field = krig_map.field.T * mask_array

    return krig_map


class StorageChangeCalculator:
    def __init__(
        units: str = "English",
        storage_coefficient: float = 0.2,
        anisotropic: str = "x",
    ):
        """
        Initialize the StorageChangeCalculator.

        :param raster: The netCDF4 dataset containing raster data.
        :type raster: nc.Dataset
        :param units: The system of measurement for imputed data. Choose between "English" and "Metric".
        :type units: str, optional
        :param storage_coefficient: The storage coefficient of the aquifer.
        :type storage_coefficient: float, optional
        """

        units = units
        storage_coefficient = storage_coefficient
        anisotropic = anisotropic
        degrees_to_radians = math.pi / 180

        if units == "English":
            unit_coeff = 1
            area_coeff = 43560  # 1 acre = 43560 ft^2
            vol_coeff = 1  # 1 acre-ft = 1 acre * 1 ft
            conversion_factor = 5280  # 1 mile = 5280 ft
            conversion_factor2 = 10**6  # 1 million = 1e6
            radius_earth = 3958.8 * conversion_factor  # radius of earth is 3958.8 miles
            calculation_unit = "feet"
            area_unit = "million acres"
            volume_unit = "million acre-feet"

        # TODO: Fix metric units
        # elif units == "Metric":
        #     unit_coeff = 0.3048  # 1 ft = 0.3048 m
        #     area_coeff = 1  # 1 km^2 = 1 km^2
        #     vol_coeff = 1000  # 1 km^3 = 1000 m^3
        #     conversion_factor = 1000  # 1 km = 1000 m
        #     conversion_factor2 = 1  # 1 = 1 (no conversion)
        #     radius_earth = (
        #         6378.1 * conversion_factor
        #     )  # radius of earth is 6371 km
        #     calculation_unit = "meters"
        #     area_unit = "km^2"
        #     volume_unit = "km^3"

        else:
            logging.error(
                "Invalid units. Please choose between 'English' and 'Metric'."
            )
            raise ValueError(
                "Invalid units. Please choose between 'English' and 'Metric'."
            )

        # Check anisotropy that the cells have the same resolution in one direction
        assert anisotropic in [
            "x",
            "y",
        ], "Invalid anisotropy. Please choose between 'x' and 'y'."
        logging.info(f"Cells have same resolution in {anisotropic} direction.")

        # Check storage coefficient
        assert (
            storage_coefficient >= 0 and storage_coefficient <= 1
        ), "The storage coefficient must be between 0 and 1."
        logging.info("The storage coefficient is set to %.2f.", storage_coefficient)

        return

    def calulate_storage_curve(
        raster: nc.Dataset,
        date_range_filter: Tuple[str, str] = None,
    ) -> pd.Series:
        """
        Calculate and plot the storage change curve. Over specified time period.

        This function calculates the storage change curve for a given aquifer based on
        imputed raster data. It computes the area of the aquifer, the drawdown at each
        time step, and plots the storage depletion curve. Part of this function is
        based on the code from the NOAA, where we need to convert from degrees to
        radians and calculate the area of the aquifer.

        :param raster: The netCDF4 dataset containing raster data.
        :type raster: nc.Dataset
        :param date_start: The start date of the time period.
        :type date_start: str
        :param date_end: The end date of the time period.
        :type date_end: str

        :return: None
        :rtype: None
        """
        logging.info("Calculating raster area...")
        area = calculate_wgs84_area(raster)
        logging.info("Area calculated.")
        logging.info(
            f"The area of the aquifer is: {round(area / conversion_factor2 / area_coeff, 2)} {area_unit}"
        )

        logging.info("Calculating monthly deltas...")
        delta_h = calculate_monthly_deltas(raster)
        logging.info("Monthly deltas calculated.")

        logging.info("Calculating storage change...")
        storage_change = calculate_volume(delta_h, area, storage_coefficient)
        logging.info("Storage change calculated.")

        if date_range_filter:
            logging.info("Filtering storage change curve...")
            storage_change = filter_timeseries(storage_change, date_range_filter)
            logging.info("Storage change curve filtered.")

        storage_change = unit_conversion(storage_change)
        logging.info(
            f"Final drawdown calculated: {round(storage_change[-1], 2)} {volume_unit}"
        )

        return storage_change

    def calculate_monthly_deltas(
        raster: nc.Dataset,
    ) -> pd.Series:
        """
        Calculate a storage change curve metric over a specified area and time range.

        This function calculates the storage change curve metric by comparing
        the water table elevation (WTE) at each time step with the WTE at the
        initial time step and averaging the drawdown across the entire aquifer
        for the specified date range.

        :param raster: The netCDF4 dataset containing raster data.
        :type raster: nc.Dataset
        :param area: The area of the aquifer.
        :type area: float
        :param date_start: The start date of the time period.
        :type date_start: str
        :param date_end: The end date of the time period.
        :type date_end: str
        :return: The storage change curve metric.
        :rtype: pd.Series
        """

        # Check that the raster has a time dimension
        check_time_dimension(raster)

        # Extract time index and validate date range
        datetime_index = extract_and_validate_time_index(
            raster,
        )

        # Extract raster values and calculate differences
        raster_difference = calculate_raster_differences(raster)

        # Create a pandas series of the raster differences
        delta_h = pd.Series(raster_difference, index=datetime_index)

        return delta_h

    def check_time_dimension(raster: nc.Dataset):
        if "time" not in raster.dimensions:
            logging.error("No time dimension found.")
            raise ValueError("No time dimension found.")

    def extract_and_validate_time_index(
        raster: nc.Dataset,
    ) -> pd.DatetimeIndex:
        """
        Extract the time index from a netCDF4 time variable.

        :param time_variable: A netCDF4 time variable.
        :type time_variable: Variable
        :param datetime_units: The units of the time variable. Must be in the format "%Y-%m-%d".
        :type datetime_units: str
        :return: A pandas DatetimeIndex. The time index of the netCDF4 time variable.
        :rtype: pd.DatetimeIndex
        """
        datetime_units = raster["time"].units
        cft_timestamps = nc.num2date(raster["time"][:], datetime_units)
        datetime_objects = [
            pd.Timestamp(dt.strftime("%Y-%m-%d %H:%M:%S")) for dt in cft_timestamps
        ]
        datetime_index = pd.DatetimeIndex(datetime_objects)
        return datetime_index

    def calculate_raster_differences(raster: nc.Dataset):
        raster_values = raster["tsvalue"][:]
        raster_difference = raster_values[:] - raster_values[0]
        raster_difference = np.nanmean(raster_difference, axis=(1, 2))
        return raster_difference

    def filter_timeseries(timeseries: pd.Series, date_range_filter: Tuple[str, str]):
        filter_start = pd.Timestamp(date_range_filter[0])
        filter_end = pd.Timestamp(date_range_filter[1])

        assert filter_start <= filter_end, "The start date must be before the end date."
        assert (
            filter_start >= timeseries.index[0]
        ), "The start date is not within the time range of the raster."
        assert (
            filter_end <= timeseries.index[-1]
        ), "The end date is not within the time range of the raster."

        # Filter the timeseries
        filtered_data = timeseries[
            (timeseries.index >= filter_start) & (timeseries.index <= filter_end)
        ]

        # Reset the index to start at 0
        filtered_data = filtered_data - filtered_data[0]
        return filtered_data

    def calculate_wgs84_area(
        raster: nc.Dataset,
    ):
        """
        Calculate the area of the aquifer.

        Loop through every cell in the raster and calculate the area of the aquifer
        based on the cell size and the latitude of the cell. This function is based
        on the code from the NOAA, where we need to convert from degrees to radians
        and calculate the area of the aquifer.

        Area calculated based on the equation found here:
        https://www.pmel.noaa.gov/maillists/tmap/ferret_users/fu_2004/msg00023.html
            A = 2 * pi * R * h
            h = R * (1 - sin(lat))
            A = 2 * pi * R^2 * (1 - sin(lat))
        Therefore the area between two lines of latitude is:
        A = 2 * pi * R^2 * |(lat1)-(lat2)|
        Converting from degrees to radians turns the equation into:
        A = 2*pi*R^2 * |sin(lat1)-sin(lat2)| * |lon1-lon2|/360
        A = (pi/180) *L * R^2 * |sin(lat1)-sin(lat2)| * |lon1-lon2|
        A = arc_length * R^2 * |sin(lat1)-sin(lat2)| * |lon1-lon2|

        :param raster: The netCDF4 dataset containing raster data.
        :type raster: nc.Dataset
        :param y_resolution: The resolution of the raster in the y-direction.
        :type y_resolution: float
        :param x_resolution: The resolution of the raster in the x-direction.
        :type x_resolution: float

        :return: The area of the aquifer.
        :rtype: float
        """

        # TODO: Works for x, but not y. Need to fix.

        # this assumes all cells will be the same size in one dimension currently x
        x_resolution = abs(round(raster["lon"][0] - raster["lon"][1], 7))  # degrees
        y_resolution = abs(round(raster["lat"][0] - raster["lat"][1], 7))  # degrees

        area = 0

        # Loop through each y row
        for y in range(raster.dimensions["lat"].size):
            # Define the upper and lower bounds of the row
            mylatmax = math.radians(raster["lat"][y] + (y_resolution / 2))
            mylatmin = math.radians(raster["lat"][y] - (y_resolution / 2))

            # Count how many cells in each row are in aquifer (i.e. and, therefore, not nan)
            num_cells = (~np.isnan(raster["tsvalue"][0, y, :])).sum()

            # Calculate the area of the row
            area += (
                (degrees_to_radians * x_resolution * num_cells)
                * (radius_earth) ** 2
                * abs((math.sin(mylatmin) - math.sin(mylatmax)))
            )

        return area

    def calculate_volume(
        timeseries: pd.Series, area: float, storage_coefficient: float
    ):
        """
        Calculate the storage change curve.

        :param timeseries: The timeseries of the storage change curve.
        :type timeseries: pd.Series
        :param area: The area of the aquifer.
        :type area: float
        :param storage_coefficient: The storage coefficient of the aquifer.
        :type storage_coefficient: float
        :return: The storage change curve.
        :rtype: pd.Series
        """

        storage_change = timeseries * area * storage_coefficient
        return storage_change

    def unit_conversion(storage_change: pd.Series):
        """
        Convert the units of the storage change curve.

        :param storage_change: The storage change curve.
        :type storage_change: pd.Series
        :return: The storage change curve with converted units.
        :rtype: pd.Series
        """

        storage_change = storage_change / conversion_factor2 / area_coeff / vol_coeff

        return storage_change
