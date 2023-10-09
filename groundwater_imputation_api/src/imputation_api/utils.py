"""Utility functions for the time series analysis of groundwater using GLDAS and PDSI data."""
import utils_data_classes as dc
import logging
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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
        THIS_DIR: str = (Path(__file__).parent.absolute(),)
        DATA_DIR: str = THIS_DIR / "Datasets"
        SHAPE_DIR: str = THIS_DIR / "Aquifer Shapes/"
        FIGURE_DIR: str = THIS_DIR / figures_dir
        directories = [self.data_dir, self.figure_dir, self.shape_dir]
        for path in directories:
            os.makedirs(path, exist_ok=True)


def date_parse(date: str) -> str:
    """
    Parse a date string.

    This function parses a date string in 'YYYYMM' format and converts it to 'YYYY-MM-01' format.

    Args:
        date (str): The date string in 'YYYYMM' format.

    Returns:
        str: The parsed date string in 'YYYY-MM-01' format.
    """
    substring = date.split(".")[1]
    year = substring[1:5]
    month = substring[5:7]
    return f"{year}-{month}-01"


def get_date_range(date_start: str, date_end: str, parse: bool = True):
    """
    Generate a date range.

    This function generates a date range between two dates in 'YYYYMM' format.

    Args:
        date_start (str): The start date in 'YYYYMM' format.
        date_end (str): The end date in 'YYYYMM' format.
        parse (bool, optional): Whether to parse the input dates. Defaults to True.

    Returns:
        pd.date_range: A pandas date range object.
    """
    if parse:
        date_start = date_parse(date_start)
        date_end = date_parse(date_end)
    dates = pd.date_range(start=date_start, end=date_end, freq="MS")
    return dates


def filter_dataframe_by_monthly_step(df:pd.DataFrame, monthly_step:int) -> pd.DataFrame:
    """
    Filter a dataframe by a monthly step.

    :param df: dataframe to filter
    :type df: pd.DataFrame
    :param monthly_step: number of months to skip
    :type monthly_step: int
    :return: filtered dataframe
    :rtype: pd.DataFrame
    """
    filtered_data = df.iloc[::monthly_step, :]
    return filtered_data


def load_shapefile(path: str) -> fiona.Collection:
    """
    Load a shapefile using fiona.

    :param path: path to the shapefile
    :type path: str
    :return: shapefile object
    :rtype: ShapeFileCollection
    """
    try:
        shape = fiona.open(path)
        return shape
    except FileNotFoundError as e:
        logger.error(f"Shapefile not found at path: {path}")
        raise e
    except fiona.errors.DriverError as e:
        logger.error(f"Error loading shapefile at path: {path}")
        raise e


def save_pickle(
    data: Dict, file_name: str, directory: str = "./", protocol: int = 3
) -> None:
    """
    Save a dictionary as a pickle file.

    :param data: Dictionary to be saved.
    :type data: dict
    :param file_name: Name of the pickle file.
    :type file_name: str
    :param directory: Directory where the file should be saved (default: current directory).
    :type directory: str
    :param protocol: Pickle protocol version (default: 3).
    :type protocol: int
    """
    try:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, file_name), "wb") as handle:
            pickle.dump(data, handle, protocol=protocol)
        logger.info(f"Pickle file '{file_name}' saved successfully to '{directory}'")
    except Exception as e:
        logger.error(f"Error saving pickle file '{file_name}' to '{directory}'")
        raise RuntimeError(f"Error saving pickle file: {e}")


def load_pickle(file_name: str, directory: str = "./") -> Dict:
    """
    Load a dictionary from a pickle file.

    :param file_name: Name of the pickle file.
    :type file_name: str
    :param directory: Directory where the file is located (default: current directory).
    :type directory: str
    :return: Loaded dictionary.
    :rtype: dict
    """
    try:
        with open(os.path.join(directory, file_name), "rb") as handle:
            data = pickle.load(handle)
        logger.info(f"Pickle file '{file_name}' loaded successfully from '{directory}'")
        return data
    except FileNotFoundError as e:
        logger.error(
            f"Pickle file not found at path: {os.path.join(directory, file_name)}"
        )
        raise FileNotFoundError(
            f"Pickle file not found at path: {os.path.join(directory, file_name)}"
        )
    except Exception as e:
        logger.error(f"Error loading pickle file: {e}")
        raise RuntimeError(f"Error loading pickle file: {e}")


def pull_relevant_data(
    shape: fiona.Collection,
    dataset_name: str = "GLDAS",
    dataset_directory: str = "./",
) -> Dict:
    """
    Pull relevant data from a dataset.

    Returns a pandas dataframe with the relevant data GLDAS or PDSI for cells that intercept a shapefile.

    :param shape: Shapefile object in WGS84
    :type shape: fiona.Collection
    :param dataset_name: "GLDAS" or "PDSI", defaults to "GLDAS"
    :type dataset_name: str, optional
    :rtype: dict
    """
    if dataset_name == "GLDAS":
        dataset = dc.GLDASData()
    elif dataset_name == "PDSI":
        dataset = dc.PDSIData()
    else:
        logging.error("Invalid dataset name.")
        raise ValueError("Invalid dataset name.")
    logger.info(f"Analyzing {dataset_name} dataset...")

    logger.info("Creating boundary polygon...")
    boundary = shapefile_boundary(shape)
    logger.info(f"Boundary polygon created: {boundary}")

    logger.info("Creating pseudo grid...")
    grid = create_grid(
        x_min=dataset.x_min,
        y_min=dataset.y_min,
        x_max=dataset.x_max,
        y_max=dataset.y_max,
        resolution_x=dataset.resolution_x,
        resolution_y=dataset.resolution_y,
    )
    logger.info("Grid created.")

    logger.info("Finding intercepting cells...")
    mask = find_intercepting_cells(
        boundary, grid, dataset.resolution_x, dataset.resolution_y
    )
    logger.info(f"Found {len(mask)} cells.")

    logger.info("Pulling data from cells...")
    data = query_cells(dataset.variables, mask, dataset_directory)
    logger.info("Data pulled.")

    logger.info("Validating data...")
    data = validate_data(data, mask)
    logger.info("Data validated.")
    print("\n")
    return data


def shapefile_boundary(shape: fiona.Collection) -> Tuple[float, float, float, float]:
    """
    Obtain the bounding box boundary of a shapefile. Must be in WGS84.

    :param shape: fiona shapefile
    :type shape: ShapeFileCollection
    :return: bounding box boundary of the shapefile as (minx, miny, maxx, maxy)
    :rtype: touple
    """
    if not isinstance(shape, fiona.Collection):
        raise TypeError("Input must be a fiona shapefile collection.")
    boundary = shape.bounds
    if not boundary:
        raise ValueError("Shapefile boundary is empty or None.")

    minx, miny, maxx, maxy = boundary

    if minx >= maxx or miny >= maxy:
        raise ValueError("Invalid shapefile boundary. Check that it's in WGS84.")

    return minx, miny, maxx, maxy


def create_grid(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    resolution_x: float,
    resolution_y: float,
) -> Dict[str, dict]:
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
    boundary: tuple,
    grid: Dict[str, dict],
    resolution_x: float,
    resolution_y: float,
    pad: bool = True,
) -> Dict[str, List[float]]:
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
    :param pad: whether or not to pad the boundary, defaults to True
    :type pad: bool, optional
    :return: list of cell names that intercept the padded shapefile boundary
    :rtype: list
    """
    # Check for valid boundary
    if not boundary or len(boundary) != 4:
        logger.error(
            "Invalid boundary. It should contain (x_min, y_min, x_max, y_max) coordinates."
        )
        raise ValueError(
            "Invalid boundary. It should contain (x_min, y_min, x_max, y_max) coordinates."
        )

    # Check for non-empty grid
    if not grid:
        logger.error(
            "Empty grid. The grid dictionary should contain valid grid locations."
        )
        raise ValueError(
            "Empty grid. The grid dictionary should contain valid grid locations."
        )

    # Check for positive padding resolution
    if resolution_x <= 0 or resolution_y <= 0:
        logger.error(
            "Invalid padding resolution. Both resolution_x and resolution_y should be positive."
        )
        raise ValueError(
            "Invalid padding resolution. Both resolution_x and resolution_y should be positive."
        )

    # Check for valid pad parameter
    if not isinstance(pad, bool):
        logger.error(
            "Invalid pad parameter. It should be a boolean value (True or False)."
        )
        raise ValueError(
            "Invalid pad parameter. It should be a boolean value (True or False)."
        )

    if pad:
        boundary_to_use = pad_boundary(boundary, resolution_x / 2, resolution_y / 2)
    else:
        boundary_to_use = pad_boundary(boundary, 0, 0)
    mask = intercepting_cells(boundary_to_use, grid)

    # Check for a valid mask
    if mask is None:
        logger.error("The mask returned by intercepting_cells is None.")
        raise ValueError("The mask returned by intercepting_cells is None.")

    # Check that the mask is not empty
    if not mask:
        logger.error("The mask is empty. No cells intercept the shapefile boundary.")
        raise ValueError(
            "The mask is empty. No cells intercept the shapefile boundary."
        )

    return mask


def pad_boundary(
    bounding_box: Tuple[float, float, float, float],
    padding_x: float = 0,
    padding_y: float = 0,
) -> Polygon:
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
    :rtype: shape: Polygon
    """
    if not isinstance(bounding_box, tuple) or len(bounding_box) != 4:
        logger.error("Bounding box must be a tuple of length 4.")
        raise ValueError("Bounding box must be a tuple of length 4.")

    if not (0 <= padding_x <= 180):
        logger.error(
            "Padding in the x (lon) direction must be between 0 and 180 degrees."
        )
        raise ValueError(
            "Padding in the x (lon) direction must be between 0 and 180 degrees."
        )

    if not (0 <= padding_y <= 90):
        logger.error(
            "Padding in the y (lat) direction must be between 0 and 90 degrees."
        )
        raise ValueError(
            "Padding in the y (lat) direction must be between 0 and 90 degrees."
        )

    minx = bounding_box[0] - padding_x
    miny = bounding_box[1] - padding_y
    maxx = bounding_box[2] + padding_x
    maxy = bounding_box[3] + padding_y

    padded_bounding_box = (minx, miny, maxx, maxy)
    polygon = shapely.geometry.box(*padded_bounding_box)

    return polygon


def intercepting_cells(
    shape: Polygon,
    grid: Dict[str, dict],
    exclude_cells: List[str] = [],
    naive_search: bool = False,
) -> Dict[str, List[float]]:
    """
    Given a shape find the cells that are contained within the shape.

    :param shape: shapely geometry of the polygon
    :type shape: Polygon
    :param grid: Dictionary of grid locations containing the latitude and longitude of each cell.
    :type grid: dict
    :param exclude_cells: List of cells to exclude from the search, defaults to []
    :type exclude_cells: List[str], optional
    :param naive_search: Whether or not to use a naive search, defaults to True
    :type naive_search: bool, optional
    :return: list of cells whose centroids are contained within the shape
    :rtype: list
    """
    if exclude_cells is None:
        exclude_cells = []

    if naive_search:
        intercepted_cells = {
            cell_name: [centroid["longitude"], centroid["latitude"]]
            for cell_name, centroid in grid.items()
            if shape.contains(
                shapely.geometry.Point(centroid["longitude"], centroid["latitude"])
            )
            and cell_name not in exclude_cells
        }
    else:
        intercepted_cells = {}
        # Extract the bounding box of the rectangular shape
        minx, miny, maxx, maxy = shape.bounds

        # Convert the coordinates to NumPy arrays for vectorized comparison
        longitudes = np.array([centroid["longitude"] for centroid in grid.values()])
        latitudes = np.array([centroid["latitude"] for centroid in grid.values()])

        # Vectorized rectangular search
        mask = (
            (minx <= longitudes)
            & (longitudes <= maxx)
            & (miny <= latitudes)
            & (latitudes <= maxy)
        )

        # Filter cells based on the mask
        for i, (cell_name, centroid) in enumerate(grid.items()):
            if mask[i] and cell_name not in exclude_cells:
                intercepted_cells[cell_name] = [
                    centroid["longitude"],
                    centroid["latitude"],
                ]

    return intercepted_cells


def query_cells(
    variables: List[str],
    mask: Dict[str, List[float]],
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
    Open and parse remote sensing data for specified variables and cells.

    :param variables: List of variables to query.
    :type variables: List[str]
    :param cells: List of cell names.
    :type cells: List[str]
    :param data_folder_path: Path to the data folder.
    :type data_folder_path: str
    :return: Dictionary of dataframes of combined variables {variable: pd.DataFrame}.
    :rtype: Dict[str, pd.DataFrame]
    """
    data_dictionary = dict()
    for _, variable in enumerate(variables):
        file_name = f"{variable}.pickle"
        df_temp = load_pickle(file_name, data_folder_path)
        data_dictionary[variable] = df_temp[cells]

    return data_dictionary


def _add_location_dataframe(mask: Dict[str, List[float]]) -> pd.DataFrame:
    """
    add location dataframe to the data dictionary

    :param cells: list of cell names
    :type cells: List[str]
    :return: dataframe of cell locations
    :rtype: pd.DataFrame
    """

    location_data = []

    for cell, (longitude, latitude) in mask.items():
        location_data.append(
            {"CellName": cell, "Longitude": longitude, "Latitude": latitude}
        )

    df_location = pd.DataFrame(location_data)

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
        logger.info(f"Parsing {cell} {i+1} / {len(cells)}")

        # Create df_cell_feature by selecting relevant variables
        df_cell_feature = dict_features[variables[0]][cell].astype(float)

        # Check if data exists (considering any non-NaN value)
        if df_cell_feature.notna().any() or not df_cell_feature.dropna().empty:
            data[cell] = pd.DataFrame(
                {
                    variable: dict_features[variable][cell].astype(float)
                    for variable in variables
                }
            )
        else:
            df_location.drop([cell], inplace=True, axis=0)

    # Add the location dataframe to the data dictionary with values for cells that exist
    data["locations"] = df_location

    return data


def validate_data(data: Dict, mask: List) -> Dict:
    """
    Validate the data by removing cells that do not have any valid columns.

    This function checks each column of each cell and removes the cell if all columns are empty.

    :param data: A dictionary containing cell names as keys and DataFrames as values.
    :type data: Dict[str, pd.DataFrame]
    :param mask: A list of cell names to validate.
    :type mask: List[str]
    :return: A dictionary with invalid cells removed.
    :rtype: Dict[str, pd.DataFrame]
    """
    invalid_cells = []

    for cell in mask:
        try:
            validated_columns = [var for var in data[cell] if not data[cell][var].empty]

            if validated_columns:
                data[cell] = data[cell][validated_columns]
            else:
                invalid_cells.append(cell)
                logger.error(f"Invalid cell: {cell}")
        except Exception as e:
            logger.error(f"Error with cell: {cell}. Cell Not Validated: {str(e)}")
            pass

    for cell in invalid_cells:
        del data[cell]

    logger.info(f"Removed {len(invalid_cells)} invalid cells.")
    logger.info(f"Remaining {len(data) - 1} cells.")

    return data


def transform_well_data(
    well_timeseries: pd.DataFrame,
    well_locations: pd.DataFrame,
    timeseries_name: str = "timeseries",
    locations_name: str = "locations",
) -> Dict[str, Union[pd.DataFrame, pd.DataFrame]]:
    """
    Transform well timeseries and locations data into a processed dictionary.

    Input dataframes must have the following columns:
    well_timeseries: ["Date", "Well ID", "Measurement"]
    well_locations: ["Well ID", "Longitude", "Latitude"]

    :param well_timeseries: Dataframe of well timeseries data.
    :type well_timeseries: pd.DataFrame
    :param well_locations: Dataframe of well locations.
    :type well_locations: pd.DataFrame
    :return: Processed well data dictionary.
    :rtype: Dict[str, Union[pd.DataFrame, pd.DataFrame]]
    """
    logger.info("Making well dictionary...")
    raw_data = make_well_dict(
        well_timeseries=well_timeseries,
        well_locations=well_locations,
        well_timeseries_name=timeseries_name,
        well_locations_name=locations_name,
    )
    logger.info("Well dictionary made.")

    logger.info("Processing well data...")
    try:
        data = process_well_data(
            timeseries=raw_data[f"{timeseries_name}_raw"],
            locations=raw_data[f"{locations_name}_raw"],
            well_timeseries_name=timeseries_name,
            well_locations_name=locations_name,
            std_threshold=3,
            min_monthly_obs=50,
            gap_size=365,
            pad=90,
            start_date="1/1/1948",
            end_date="1/1/2020",
        )
    except Exception as e:
        logger.error(f"Error processing well data: {e}")
        raise RuntimeError(f"Error processing well data: {e}")
    logger.info("Well data processed.")

    logger.info("Merging well data...")
    data = merge_dictionaries(data, raw_data)
    logger.info("Well data merged.")

    return data


def make_well_dict(
    well_timeseries: pd.DataFrame,
    well_locations: pd.DataFrame,
    well_timeseries_name: str = "timeseries",
    well_locations_name: str = "locations",
) -> Dict[str, Union[pd.DataFrame, pd.DataFrame]]:
    """
    Feed in the well timeseries and locations and return a dictionary of well data.
    Name is appended with "_raw" to indicate that the data is raw and has not been processed.

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
    assert all(
        col in well_timeseries.columns for col in ["Date", "Well ID", "Measurement"]
    ), "Invalid well timeseries columns."
    assert all(
        col in well_locations.columns for col in ["Well ID", "Longitude", "Latitude"]
    ), "Invalid well locations columns."

    well_dict = {}
    well_dict[f"{well_locations_name}_raw"] = get_well_locations(df=well_locations)
    well_dict[f"{well_timeseries_name}_raw"] = get_well_timeseries(df=well_timeseries)
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
    expected_cols = ["Well ID", "Longitude", "Latitude"]
    assert all(
        col in df.columns for col in expected_cols
    ), f"Missing or incorrect columns in dataframe. Expected columns: {expected_cols}"

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
    expected_cols = ["Date", "Well ID", "Measurement"]
    assert all(
        col in df.columns for col in expected_cols
    ), f"Missing or incorrect columns in dataframe. Expected columns: {expected_cols}"

    df["Well ID"] = df["Well ID"].astype(str)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    assert not df["Date"].isna().any(), "Invalid date format."

    try:
        df["Measurement"] = df["Measurement"].astype(float)
    except ValueError:
        logger.warning("Measurement column contains non-numeric values.")
        raise ValueError("Measurement column contains non-numeric values.")

    # Pivot the dataframe to have "Well ID" as columns and "Date" as index
    df = df.pivot(index="Date", columns="Well ID", values="Measurement")
    return df


def process_well_data(
    timeseries: pd.DataFrame,
    locations: pd.DataFrame,
    well_timeseries_name: str = "timeseries",
    well_locations_name: str = "locations",
    std_threshold: int = 3,
    min_monthly_obs: int = 50,
    gap_size: int = 365,
    pad: int = 90,
    start_date: str = "1/1/1948",
    end_date: str = "1/1/2020",
) -> Dict[str, Union[pd.DataFrame, pd.DataFrame]]:
    """
    Process well timeseries and locations data.

    :param timeseries: Dataframe of well timeseries data.
    :type timeseries: pd.DataFrame
    :param locations: Dataframe of well locations.
    :type locations: pd.DataFrame
    :param well_timeseries_name: Name for well timeseries data in the output dictionary.
    :type well_timeseries_name: str, optional
    :param well_locations_name: Name for well locations data in the output dictionary.
    :type well_locations_name: str, optional
    :param std_threshold: Standard deviation threshold for outlier removal, defaults to 3.
    :type std_threshold: int, optional
    :param min_monthly_obs: Minimum monthly observations for well selection, defaults to 50.
    :type min_monthly_obs: int, optional
    :param gap_size: Gap size for padding, defaults to 365.
    :type gap_size: int, optional
    :param pad: Padding size for padding, defaults to 90.
    :type pad: int, optional
    :param start_date: Start date for data filtering, defaults to "1/1/1948".
    :type start_date: str, optional
    :param end_date: End date for data filtering, defaults to "1/1/2020".
    :type end_date: str, optional
    :return: Processed well data dictionary.
    :rtype: Dict[str, Union[pd.DataFrame, pd.DataFrame]]
    """

    logger.info("Removing outliers...")
    timeseries = remove_outliers(timeseries=timeseries, std_threshold=std_threshold)
    logger.info("Outliers removed.")

    logger.info("Selecting qualifing wells...")
    timeseries = select_qualifing_wells(
        timeseries=timeseries,
        start_date=start_date,
        end_date=end_date,
        min_monthly_obs=min_monthly_obs,
    )
    logger.info("Qualifing wells selected.")

    logger.info("Padding well measurements...")
    timeseries = pad_wells(
        timeseries=timeseries,
        gap_size=gap_size,
        pad=pad,
    )
    logger.info("Well measurements padded.")

    logger.info("Updating locations...")
    locations = update_locations(
        locations=locations, names=timeseries.columns.to_list()
    )
    logger.info("Locations updated.")
    data = {well_timeseries_name: timeseries, well_locations_name: locations}
    return data


def remove_outliers(timeseries: pd.DataFrame, std_threshold: float = 3) -> pd.DataFrame:
    """
    Remove outliers from a dataframe of timeseries data.

    This function calculates lower and upper bounds for each column based on the mean and standard deviation.
    Any values outside of these bounds are replaced with NaN.

    :param timeseries: Dataframe of timeseries data.
    :type timeseries: pd.DataFrame
    :param std_threshold: Number of standard deviations from the mean to consider as outliers, defaults to 3.
    :type std_threshold: float, optional
    :return: Dataframe with outliers replaced by NaN.
    :rtype: pd.DataFrame
    """
    # Input validation
    assert isinstance(
        timeseries, pd.DataFrame
    ), "Input 'timeseries' must be a pandas DataFrame."
    assert isinstance(
        std_threshold, (int, float)
    ), "Input 'std_threshold' must be a number."
    assert std_threshold > 0, "Input 'std_threshold' must be greater than 0."

    df = timeseries.copy()
    col_mean = df.mean()
    col_std = df.std()
    lower_bounds = col_mean - (col_std * std_threshold)
    upper_bounds = col_mean + (col_std * std_threshold)
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
    """
    Create a DatetimeIndex for interpolation within a specified date range.

    :param start_date: Start date for the index in the format 'MM/DD/YYYY', defaults to '1/1/1948'.
    :type start_date: str, optional
    :param end_date: End date for the index in the format 'MM/DD/YYYY', defaults to '1/1/2020'.
    :type end_date: str, optional
    :param freq: Frequency of the index, e.g., 'MS' for monthly, defaults to 'MS'.
    :type freq: str, optional
    :return: DatetimeIndex for interpolation.
    :rtype: pd.DatetimeIndex
    """

    index = pd.date_range(start=start_date, end=end_date, freq=freq)
    return index


def mask_dataframe(
    timeseries: pd.DataFrame,
    start_date: pd.DatetimeIndex,
    end_date: pd.DatetimeIndex,
    min_monthly_obs=50,
) -> pd.DataFrame:
    """
    Mask a timeseries DataFrame based on date range and minimum monthly observations.

    :param timeseries: DataFrame containing time-series data.
    :type timeseries: pd.DataFrame
    :param start_date: Start date for the mask.
    :type start_date: pd.Timestamp
    :param end_date: End date for the mask.
    :type end_date: pd.Timestamp
    :param min_monthly_obs: Minimum number of monthly observations to retain a column, defaults to 50.
    :type min_monthly_obs: int, optional
    :return: Masked DataFrame.
    :rtype: pd.DataFrame
    """
    # Filter the DataFrame based on the date range
    date_mask = (timeseries.index >= start_date) & (timeseries.index <= end_date)
    timeseries = timeseries.loc[date_mask]

    # Calculate the count of unique months for each column using vectorized operations
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
    """
    Pad well time series data to fill gaps and remove extrapolated values.

    :param timeseries: Input DataFrame containing well time series data.
    :type timeseries: pd.DataFrame
    :param gap_size: Maximum gap size (in days) to consider as a gap, defaults to 365.
    :type gap_size: int, optional
    :param pad: Number of NaN values to pad in gaps, defaults to 90.
    :type pad: int, optional
    :param spacing: Frequency for interpolation, defaults to "1MS" (1 month).
    :type spacing: str, optional
    :return: DataFrame with padded and cleaned well time series data.
    :rtype: pd.DataFrame
    """

    df = pd.DataFrame(
        index=make_interpolation_index(
            timeseries.index[0], timeseries.index[-1], spacing
        )
    )
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
    """
    Create a mask to filter a DataFrame based on a date range.

    :param index: Datetime index of the DataFrame.
    :type index: pd.DatetimeIndex
    :param start_date: Start date of the desired date range (inclusive).
    :type start_date: pd.Timestamp
    :param end_date: End date of the desired date range (inclusive).
    :type end_date: pd.Timestamp
    :return: Boolean mask for filtering the DataFrame.
    :rtype: pd.DatetimeIndex
    """

    mask = (index >= start_date) & (index <= end_date)
    return mask


def fill_gaps_with_nans(
    series: pd.Series,
    gaps: list,
    pad: int = 90,
) -> pd.Series:
    """
    Fill gaps in a time series with NaN values.

    This function takes a time series, a list of gap intervals, and a padding
    value. It fills the specified gaps in the time series with NaN values.

    :param series: Time series data as a pandas Series.
    :type series: pd.Series
    :param gaps: List of gap intervals to be filled with NaN values.
                 Each gap is represented as a tuple (start, end).
    :type gaps: list
    :param pad: Number of days to pad the gaps before and after filling with NaN values.
                Defaults to 90.
    :type pad: int, optional
    :return: Time series with gaps filled as NaN values.
    :rtype: pd.Series
    """

    for gap in gaps:
        start = gap[0] + pd.Timedelta(days=pad)
        end = gap[1] - pd.Timedelta(days=pad)
        mask = make_dataframe_mask(series.index, start, end)
        series.loc[mask] = np.nan
    return series


def remove_extrapolated_values(
    series: pd.Series,
    range: tuple,
) -> pd.Series:
    """
    Remove extrapolated values from a time series outside of a specified date range.

    This function takes a time series and removes values that fall outside of
    the specified date range by setting them to NaN.

    :param series: Time series data as a pandas Series.
    :type series: pd.Series
    :param date_range: Tuple representing the valid date range (start_date, end_date).
    :type date_range: tuple
    :return: Time series with extrapolated values set to NaN.
    :rtype: pd.Series
    """
    mask = make_dataframe_mask(series.index, range[0], range[1])
    series.loc[~mask] = np.nan
    return series


def interpolate_pchip(
    series: pd.Series,
    interp_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Perform Piecewise Cubic Hermite Interpolation (PCHIP) on a time series.

    This function interpolates a time series using the Piecewise Cubic Hermite
    Interpolation (PCHIP) method.

    :param series: Time series data as a pandas Series.
    :type series: pd.Series
    :param interp_index: Index for the interpolated series.
    :type interp_index: pd.DatetimeIndex
    :return: Interpolated time series.
    :rtype: pd.Series
    """

    # Convert timestamps to Julian dates for interpolation
    series_julian = series.index.to_julian_date()
    interp_julian = interp_index.to_julian_date()

    # Perform PCHIP interpolation
    fit = pchip(series_julian, series.values)
    series_interp_values = fit(interp_julian)

    # Create a new Series with interpolated values and the specified index
    series_interp = pd.Series(series_interp_values, index=interp_index)

    return series_interp


def update_locations(locations: pd.DataFrame, names: list) -> pd.DataFrame:
    """
    Update the locations DataFrame to include only specific well names.

    Given a DataFrame of well locations and a list of well names to include,
    this function filters the DataFrame to keep only the specified well names.

    :param locations: DataFrame of well locations.
    :type locations: pd.DataFrame
    :param names: List of well names to include in the updated DataFrame.
    :type names: list
    :return: Updated DataFrame with specified well names.
    :rtype: pd.DataFrame
    """
    # Filter the locations DataFrame to include only specific well names
    updated_locations = locations[locations["Well ID"].isin(names)]

    # Set the "Well ID" column as the index
    updated_locations = updated_locations.set_index("Well ID")

    return updated_locations


def merge_dictionaries(dict1: dict, dict2: dict) -> dict:
    """
    Merge two dictionaries into a new dictionary.

    :param dict1: The first dictionary.
    :type dict1: dict
    :param dict2: The second dictionary.
    :type dict2: dict
    :return: A new dictionary containing key-value pairs from both input dictionaries.
    :rtype: dict
    """
    merged_dict = {**dict1, **dict2}
    return merged_dict
