"""Utility functions for the time series analysis of groundwater using GLDAS and PDSI data."""
import data_classes as dc
import numpy as np
import pandas as pd
import fiona
import shapely
import os
import pickle
from scipy.interpolate import pchip
import logging
from typing import TypeVar, Union, List, Tuple, Dict
from pathlib import Path
from fiona.collection import Collection
from shapely.geometry.polygon import Polygon

ShapeFileCollection = TypeVar("ShapeFileCollection", bound=Collection)
BaseGeometry = TypeVar("BaseGeometry", bound=shapely.geometry.base.BaseGeometry)
Polygon = TypeVar("Polygon", bound=Polygon)


class ProjectSettings:
    def __init__(
        self,
        aquifer_name: str = None,
        figures_dir: str = "Figures",
        iteration: int = 1,
    ):
        self.aquifer_name = aquifer_name
        self.iteration = iteration
        THIS_DIR: str = Path(__file__).parent.absolute(),
        DATA_DIR: str = THIS_DIR / "Datasets"
        SHAPE_DIR: str = THIS_DIR / "Aquifer Shapes/"
        FIGURE_DIR: str = THIS_DIR / figures_dir
        directories = [self.data_dir, self.figure_dir, self.shape_dir]
        for path in directories:
            os.makedirs(path, exist_ok=True)


def load_shapefile(path: str) -> ShapeFileCollection:
    """
    Load a shapefile using fiona.

    :param path: path to the shapefile
    :type path: str
    :return: shapefile object
    :rtype: ShapeFileCollection
    """
    shape = fiona.open(path)
    return shape


def save_pickle(
    data: dict, file_name: str, path: str = "./", protocol: int = 3
) -> None:
    with open(os.path.join(path, file_name), "wb") as handle:
        pickle.dump(data, handle, protocol=protocol)


def load_pickle(file_name: str, path: str = "./") -> Dict:
    with open(os.path.join(path, file_name), "rb") as handle:
        data = pickle.load(handle)
    return data


def pull_relevant_data(
    shape: ShapeFileCollection,
    dataset_name: str = "GLDAS",
    dataset_path: str = "./",
) -> Dict:
    """
    Pull relevant data from a dataset.

    Returns a pandas dataframe with the relevant data GLDAS or PDSI for cells that intercept a shapefile.

    :param shape: Shapefile object in WGS84
    :type shape: ShapeFileCollection
    :param dataset_name: "GLDAS" or "PDSI", defaults to "GLDAS"
    :type dataset_name: str, optional
    :rtype: dict
    """
    if dataset_name == "GLDAS":
        dataset = dc.GLDASData()
    elif dataset_name == "PDSI":
        dataset = dc.PDSIData()
    else:
        raise ValueError("Invalid dataset name.")
    print(f"Analyzing {dataset_name} dataset...")

    print("Creating boundary polygon...")
    boundary = shapefile_boundary(shape)
    print("Boundary polygon created.")

    print("Creating pseudo grid...")
    grid = create_grid(
        x_min=dataset.x_min,
        y_min=dataset.y_min,
        x_max=dataset.x_max,
        y_max=dataset.y_max,
        resolution_x=dataset.resolution_x,
        resolution_y=dataset.resolution_y,
    )
    print("Grid created.")

    print("Finding intercepting cells...")
    mask = find_intercepting_cells(
        boundary, grid, dataset.resolution_x, dataset.resolution_y
    )
    print(f"Found {len(mask)} cells.")

    print("Pulling data from cells...")
    data = query_cells(dataset.variables, mask, dataset_path)
    print("Data pulled.")

    print("Validating data...")
    data = validate_data(data, mask)
    print("Data validated.")
    print("\n")
    return data


def shapefile_boundary(shape: ShapeFileCollection) -> tuple:
    """
    Obtain the bounding box boundary of a shapefile. Must be in WGS84.

    :param shape: fiona shapefile
    :type shape: ShapeFileCollection
    :return: bounding box boundary of the shapefile as (minx, miny, maxx, maxy)
    :rtype: touple
    """
    boundary = shape.bounds
    return boundary


def create_grid(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    resolution_x: float,
    resolution_y: float,
) -> Dict:
    """
    Creates a grid of latitudes and longitudes for a netCDF file.

    GLDAS data is usually provided by giving the North East and South West corners of the grid.
    This function creates a grid of latitudes and longitudes for a netCDF file.

    :param north_east_corner_lat: The latitude of the north east corner center of the grid.
    :type north_east_corner_lat: float
    :param south_west_corner_lat: The latitude of the south west corner center of the grid.
    :type south_west_corner_lat: float
    :param north_east_corner_lon: The longitude of the north east corner center of the grid.
    :type north_east_corner_lon: float
    :param south_west_corner_lon: The longitude of the south west corner center of the grid.
    :type south_west_corner_lon: float
    :param resolution_x:The resolution of the grid in the x direction.
    :type resolution_x: float
    :param resolution_y: The resolution of the grid in the y direction.
    :type resolution_y: float
    :return: Dictionary of grid locations.
    :rtype: dict
    """

    # latitude range (rows): from north (90) to south (-90)
    lat_range = np.arange(y_max, y_min - resolution_y, -resolution_y)
    # longitude range (columns): from west (-180) to east (180)
    lon_range = np.arange(x_min, x_max + resolution_x, resolution_x)

    # create grid from north (90) to south (-90)
    lat_mesh, lon_mesh = np.meshgrid(lat_range, lon_range, indexing="ij")

    # create a dictionary of grid locations
    grid_locations = {
        f"Cell_{i}": {
            "latitude": lat_mesh[row_col_index],
            "longitude": lon_mesh[row_col_index],
            "row_index": row_col_index[0],
            "column_index": row_col_index[1],
        }
        for i, row_col_index in enumerate(np.ndindex(lon_mesh.shape))
    }
    return grid_locations


def find_intercepting_cells(
    boundary: tuple, grid: dict, resolution_x: float, resolution_y: float
) -> List:
    """
    Find the cells that intercept a shapefile.

    Padding is used to capture cells that are not fully contained within the shapefile.

    :param boundary: (x_min, y_min, x_max, y_max) in lat/lon coordinates WGS84
    :type boundary: tuple
    :param grid: dictionary of grid locations as returned by create_grid
    :type grid: dict
    :param resolution_x: padding in the x (lon) direction
    :type resolution_x: float
    :param resolution_y: padding in the y (lat) direction
    :type resolution_y: float
    :return: list of cell names that intercept the padded shapefile boundary
    :rtype: list
    """
    padded_boundary = pad_boundary(boundary, resolution_x / 2, resolution_y / 2)
    mask = intercepting_cells(padded_boundary, grid)
    # TODO: Add a check to make sure that the mask is not empty
    return mask


def pad_boundary(
    bounding_box: tuple, padding_x: float = 0, padding_y: float = 0
) -> BaseGeometry:
    """
    Padd the bounding box of a shapefile.

    Using bounding box because it is faster than using the shapefile itself.
    Padding is also used to capture cells that are not fully contained within the shapefile.

    :param bounding_box: bounding box of the shapefile as (minx, miny, maxx, maxy)
    :type bounding_box: tuple
    :param padding_x: Amount of padding in the x (lon) direction, defaults to 0
    :type padding_x: float, optional
    :param padding_y: Amount of padding in the y (lat) direction, defaults to 0
    :type padding_y: float, optional
    :return: shapely polygon of the padded bounding box
    :rtype: shape: BaseGeometry
    """
    assert padding_x >= 0, "Padding must be a positive number."
    assert padding_y >= 0, "Padding must be a positive number."
    assert padding_x <= 180, "Padding must be less than 180 degrees."
    assert padding_y <= 90, "Padding must be less than 90 degrees."
    assert len(bounding_box) == 4, "Bounding box must be a touple of length 4."
    minx = bounding_box[0] - padding_x
    miny = bounding_box[1] - padding_y
    maxx = bounding_box[2] + padding_x
    maxy = bounding_box[3] + padding_y
    padded_bounding_box = (minx, miny, maxx, maxy)
    polygon = shapely.geometry.box(*padded_bounding_box)
    return polygon


def intercepting_cells(
    shape: Polygon,
    grid: dict,
) -> Dict[str, List[float]]:
    """
    Given a shape find the cells that are contained within the shape.

    :param shape: shapely geometry of the polygon
    :type shape: Polygon
    :param grid: Dictionary of grid locations containing the latitude and longitude of each cell.
    :type grid: dict
    :return: list of cells whose centroids are contained within the shape
    :rtype: list
    """
    cells = {}
    for cell, centroid in grid.items():
        long, lat = centroid["longitude"], centroid["latitude"]
        point = shapely.geometry.Point(long, lat)
        if shape.contains(point):
            cells[cell] = [long, lat]
    return cells


def query_cells(
    variables: List[str],
    mask: Dict[str, tuple],
    path: str = "./",
) -> Dict[str, pd.DataFrame]:
    """
    Query the data for each cell in the mask.

    :param mask: list of cell names
    :type mask: List
    :param variables: list of variables to query
    :type variables: List
    :param path: path to the data
    :type path: str
    :return: dictionary of dataframes
    :rtype: Dict[str, pd.DataFrame]
    """
    # get the cell names
    cells = mask.keys()

    # open variable pickle files and parse into a dictionary of dataframes of variables
    dict_variables = _open_remote_sensing_data(variables, cells, path)

    # add location data to the data dictionary
    df_location = _add_location_dataframe(mask)

    # parse each variable into a dictionary of dataframes of the form {variable: {cell: data}}
    data_cells_dict = _parse_data(dict_variables, cells, variables, df_location)
    return data_cells_dict


def _open_remote_sensing_data(
    variables: List[str], cells: List[str], data_folder_path: str
) -> Dict[str, pd.DataFrame]:
    """
    function opens each variable pickle file and parses on the cells of interest

    :param variables: list of variables to query
    :type variables: List[str]
    :param cells:  list of cell names
    :type cells: List
    :param data_folder_path: path to the data
    :type data_folder_path: str
    :return: dictionary of dataframes of combined variables {variable: {pd.DataFrame}}
    :rtype: Dict[str, pd.DataFrame]
    """
    data_dictionary = dict()
    for i, variable in enumerate(variables):
        file_name = f"{variable}.pickle"
        df_temp = load_pickle(file_name, data_folder_path)
        data_dictionary[variable] = df_temp[cells]
        del df_temp
    return data_dictionary


def _add_location_dataframe(mask: Dict[str, List[float]]) -> pd.DataFrame:
    """
    add location dataframe to the data dictionary

    :param cells: list of cell names
    :type cells: List[str]
    :return: dataframe of cell locations
    :rtype: pd.DataFrame
    """
    # Convert the mask dictionary to a dictionary with column-wise data
    location_dict = {
        cell: {"Longitude": location[0], "Latitude": location[1]}
        for cell, location in mask.items()
    }
    # TODO: flip the order of the dictionary so that it is row-wise data

    # Create the DataFrame using from_dict()
    df_location = pd.DataFrame.from_dict(location_dict, orient="index")

    return df_location


def _parse_data(
    dict_features: Dict[str, pd.DataFrame],
    cells: List[str],
    variables: List[str],
    df_location: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Parse the data into a dictionary of dataframes.

    Key for each dataframe is the cell name.

    :param df_features: Dictionary of dataframes of combined variables {variable: {pd.DataFrame}}
    :type df_features: Dict[str, pd.DataFrame]
    :param cells: list of cell names
    :type cells: List[str]
    :param variables: list of variables to query
    :type variables: List[str]
    :param df_location: dataframe of cell locations
    :type df_location: pd.DataFrame
    :return: dictionary of dataframes of the form {cell: {variable: data}}
    :rtype: Dict[str, pd.DataFrame]
    """
    # create a dictionary to store the data
    data = {}
    for i, cell in enumerate(cells):
        print(f"Parsing {cell} {i+1} / {len(cells)}")

        # Create df_cell_feature using list comprehension
        df_cell_feature = pd.DataFrame(
            {
                variable: dict_features[variable][cell].astype(float)
                for variable in variables
            }
        )

        # Check if data exists, considering both mean or empty
        if (
            df_cell_feature.stack().mean() != -9999.0
            or not df_cell_feature.dropna(axis=0, how="any").empty
        ):
            data[cell] = df_cell_feature.astype(float)
        else:
            df_location.drop([cell], inplace=True, axis=0)

    # add the location dataframe to the data dictionary with values for cells that exist
    data["locations"] = df_location
    return data


def validate_data(data: Dict, mask: List) -> Dict:
    """
    Validate the data by removing cells that do not have any valid columns.

    Check each column of each cell and remove the cell if all columns are empty.

    :param data: data dictionary
    :type data: Dict
    :param mask: list of cell names
    :type mask: List
    :return: data dictionary with invalid cells removed
    :rtype: Dict
    """
    invalid_cells = []

    for cell in mask:
        try:
            validated_columns = []
            for var in data[cell].columns:
                data_temp = data[cell][var].astype(float).dropna(axis=0, how="any")
                if not data_temp.empty:
                    validated_columns.append(var)

            if validated_columns:
                data[cell] = data[cell][validated_columns]
            else:
                invalid_cells.append(cell)
                print(f"Invalid cell:  {cell}")
        except Exception as e:
            print(f"Error with cell: {cell}. Cell Not Validated.")
            pass

    for cell in invalid_cells:
        del data[cell]

    return data


def make_well_dict(
    well_timeseries: pd.DataFrame,
    well_locations: pd.DataFrame,
) -> Dict:
    """
    Feed in the well timeseries and locations and return a dictionary of well data.

    Input dataframes must have the following columns:
    well_timeseries: ["Date", "Well ID", "Measurement"]
    well_locations: ["Well ID", "Longitude", "Latitude"]

    :param well_timeseries: Dataframe of well timeseries data.
    :type well_timeseries: pd.DataFrame
    :param well_locations: Dataframe of well locations.
    :type well_locations: pd.DataFrame
    :return: Dictionary of well data formatted for pre-processing.
    :rtype: dict
    """
    well_dict = {}
    well_dict["locations_raw"] = get_well_locations(df=well_locations)
    well_dict["timeseries_raw"] = get_well_timeseries(df=well_timeseries)
    return well_dict


def get_well_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format the well locations dataframe.

    Columns must be: ["Date", "Well ID", "Measurement"]

    :param df: Dataframe
    :type df: pd.DataFrame
    :return: Dataframe
    :rtype: pd.DataFrame
    """
    assert list(map(str, df.columns)) == ["Well ID", "Longitude", "Latitude"]
    df["Well ID"] = df["Well ID"].astype(str)
    df["Longitude"] = df["Longitude"].astype(float)
    df["Latitude"] = df["Latitude"].astype(float)
    return df


def get_well_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format the well timeseries dataframe.

    Columns must be: ["Date", "Well ID", "Measurement"]

    :param df: Dataframe
    :type df: pd.DataFrame
    :return: Dataframe
    :rtype: pd.DataFrame
    """
    assert list(map(str, df.columns)) == ["Date", "Well ID", "Measurement"]
    df["Date"] = pd.to_datetime(df["Date"])
    df["Well ID"] = df["Well ID"].astype(str)
    df["Measurement"] = df["Measurement"].astype(float)
    df = df.pivot(index="Date", columns="Well ID", values="Measurement")
    return df


def process_well_data(
    timeseries: pd.DataFrame,
    locations: pd.DataFrame,
    std: int = 3,
    min_monthly_obs: int = 50,
    gap_size: int = 365,
    pad: int = 90,
    start_date: str = "1/1/1948",
    end_date: str = "1/1/2020",
) -> Dict:
    print("Removing outliers...")
    timeseries = remove_outliers(timeseries=timeseries, std=std)
    print("Outliers removed.")

    print("Selecting qualifing wells...")
    timeseries = select_qualifing_wells(
        timeseries=timeseries,
        start_date=start_date,
        end_date=end_date,
        min_monthly_obs=min_monthly_obs,
    )
    print("Qualifing wells selected.")

    print("Padding well measurements...")
    timeseries = pad_wells(
        timeseries=timeseries,
        gap_size=gap_size,
        pad=pad,
    )
    print("Well measurements padded.")

    print("Updating locations...")
    locations = update_locations(
        locations=locations, names=timeseries.columns.to_list()
    )
    print("Locations updated.")
    data = {"timeseries": timeseries, "locations": locations}
    return data


def remove_outliers(timeseries: pd.DataFrame, std: float = 3) -> pd.DataFrame:
    """
    Recieves a dataframe of timeseries data and removes outliers.

    Calculates a lower and upper bound for each column based on the mean and standard deviation.
    Any values outside of the bounds are replaced with NaN.

    :param timeseries: Dataframe of timeseries data.
    :type timeseries: pd.DataFrame
    :param std: _description_, defaults to 3
    :type std: float, optional
    :return: Dataframe without outliers.
    :rtype: pd.DataFrame
    """

    df = timeseries.copy()
    col_mean = df.mean()
    col_std = df.std()
    lower_bounds = col_mean - (col_std * std)
    upper_bounds = col_mean + (col_std * std)
    df_mask = np.logical_or(df < lower_bounds, df > upper_bounds)
    df[df_mask] = np.nan
    return df


def select_qualifing_wells(
    timeseries: pd.DataFrame,
    start_date="1/1/1948",
    end_date="1/1/2020",
    min_monthly_obs: int = 50,
) -> pd.DataFrame:
    """
    Given a dataframe of timeseries data, return only the wells that meet the selection criteria.

    Criteria: Must have at least 1 observation in 50 unique months.

    :param timeseries: Dataframe of timeseries data.
    :type timeseries: pd.DataFrame
    :param start_date: _description_, defaults to "1/1/1948"
    :type start_date: str, optional
    :param end_date: _description_, defaults to "1/1/2020"
    :type end_date: str, optional
    :param min_monthly_obs: _description_, defaults to 50
    :type min_monthly_obs: int, optional
    :return: _description_
    :rtype: pd.DataFrame
    """
    dt_index = make_interpolation_index(start_date=start_date, end_date=end_date)
    series_range = (dt_index[0], dt_index[-1])
    selected_timeseries = mask_dataframe(
        timeseries, series_range[0], series_range[1], min_monthly_obs
    )
    timeseries = timeseries[selected_timeseries.columns]
    return timeseries


def make_interpolation_index(
    start_date: str = "1/1/1948", end_date: str = "1/1/2020", freq: str = "MS"
) -> pd.DatetimeIndex:
    index = pd.date_range(start=start_date, end=end_date, freq=freq)
    return index


def mask_dataframe(
    timeseries: pd.DataFrame,
    start_date: pd.DatetimeIndex,
    end_date: pd.DatetimeIndex,
    min_monthly_obs=50,
) -> pd.DataFrame:
    mask = (timeseries.index >= start_date) & (timeseries.index <= end_date)
    timeseries = timeseries.loc[mask]
    timeseries = timeseries.drop(
        timeseries.columns[
            timeseries.apply(
                lambda col: len(
                    np.unique((col.dropna().index).strftime("%Y/%m")).tolist()
                )
                < min_monthly_obs
            )
        ],
        axis=1,
    )
    return timeseries


def pad_wells(
    timeseries: pd.DataFrame,
    gap_size: int = 365,
    pad: int = 90,
    spacing: str = "1MS",
):
    df = pd.DataFrame(index=make_interpolation_index())
    for well_id in timeseries:
        series = timeseries[well_id].dropna()
        series_range = (series.index[0], series.index[-1])
        interp_index = make_interpolation_index(
            start_date=series_range[0], end_date=series_range[1], freq=spacing
        )
        series_interp = interpolate_pchip(series, interp_index)
        # grab the difference between each index value
        index_delta = (series.index[1:] - series.index[:-1]).days
        # find the gaps that are larger than the gap size
        gap_indecies = np.where(index_delta > gap_size)[0]
        # get the start and end of each gap
        gaps = [(series.index[i], series.index[i + 1]) for i in gap_indecies]
        # remove interpolated values from gaps larger than gap_size
        series = fill_gaps_with_nans(series_interp, gaps, pad)
        series = remove_extrapolated_values(series, series_range)
        series.name = str(well_id)
        df = pd.concat([df, series], join="outer", axis=1, sort=True)
    return df


def make_dataframe_mask(
    index: pd.DatetimeIndex,
    start_date: pd.DatetimeIndex,
    end_date: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    mask = (index >= start_date) & (index <= end_date)
    return mask


def fill_gaps_with_nans(series: pd.Series, gaps: list, pad: int = 90):
    for gap in gaps:
        start = gap[0] + pd.Timedelta(days=pad)
        end = gap[1] - pd.Timedelta(days=pad)
        mask = make_dataframe_mask(series.index, start, end)
        series.loc[mask] = np.nan
    return series


def remove_extrapolated_values(series: pd.Series, range: tuple):
    mask = make_dataframe_mask(series.index, range[0], range[1])
    series.loc[~mask] = np.nan
    return series


def interpolate_pchip(series: pd.Series, interp_index: pd.DatetimeIndex) -> pd.Series:
    fit = pchip(series.index.to_julian_date(), series.values)
    series_interp = fit(interp_index.to_julian_date())
    series_interp = pd.Series(series_interp, index=interp_index)
    return series_interp


def update_locations(locations: pd.DataFrame, names: list) -> pd.DataFrame:
    locations = locations[locations["Well ID"].isin(names)]
    locations = locations.set_index("Well ID")
    return locations


def merge_dictionaries(dict1: dict, dict2: dict) -> dict:
    dict3 = {**dict1, **dict2}
    return dict3


if __name__ == "__main__":
    path_shape = ".\Aquifer Shapes\Beryl_Enterprise.shp"
    aquifer_shape = load_shapefile(path_shape)

    path_pdsi = r"C:\Users\saulg\Desktop\Remote_Data\pdsi_tabular"
    pdsi = pull_relevant_data(
        aquifer_shape, dataset_name="PDSI", dataset_path=path_pdsi
    )
    print("done")
