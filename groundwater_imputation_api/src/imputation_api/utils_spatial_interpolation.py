import utils
import logging
import fiona
import numpy as np
import pandas as pd
import netCDF4 as nc
import gstools as gs

from typing import Tuple, Union, List
from shapely.geometry import shape, Point
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
    influence_distance: float = 0.125,
    monthly_time_step: int = 1,
    netcdf_filename: str = "interpolation.nc",
    directory: str = "./",
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
        raster_data[i, :, :] = surface.field
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

    # TODO: add ability to close the NetCDF file

    return krig_map
