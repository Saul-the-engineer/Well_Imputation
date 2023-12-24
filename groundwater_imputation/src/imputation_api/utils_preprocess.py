import os
import utils
import utils_data_classes
import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime

from tqdm import tqdm


def process_pdsi_data(
    source_directory: str,
    target_directory: str,
    date_start="01/01/1850",
    date_end="12/31/2020",
) -> None:
    """
    Convert PDSI data from netCDF4 format to pandas dataframe format.

    Saves the data as a pickle file.

    :param source_directory: location of the netCDF4 files
    :type source_directory: str
    :param target_directory: location to save the pandas dataframe files
    :type target_directory: str
    :param date_start: Date index begining for netCDF, defaults to "01/01/1850"
    :type date_start: str, optional
    :param date_end: Date index end for netCDF, defaults to "12/31/2020"
    :type date_end: str, optional
    """
    # Step 1: Get date range
    date_format = "%m/%d/%Y"
    date_start = datetime.strptime(date_start, date_format)
    date_end = datetime.strptime(date_end, date_format)
    dates = utils.get_date_range(date_start=date_start, date_end=date_end, parse=False)

    pdsi = utils_data_classes.PDSIData()
    variables = pdsi.variables
    files = pdsi.get_file_names()
    cell_names = ["Cell_" + str(i) for i in range(pdsi.n_cells)]
    data_collection = {variable: [] for variable in variables}

    for i, variable in enumerate(variables):
        # step 1: open up the dataset, parse the array and flip it so that the orientation is correct
        for j, file in tqdm(enumerate(files)):
            dataset = nc.Dataset(os.path.join(source_directory, file))
            array = np.flip(dataset.variables[variable][:].data, axis=1)
            data_collection[variable].append(array)

        # step 2: squeeze arrays to make them two-dimensional
        variable_array = np.squeeze(np.array(data_collection[variable]), axis=0)

        # step 3: flatten array by rows, then reshape it based on time (rows) and cells (columns)
        flat_variable_array = variable_array.reshape(-1, pdsi.n_cells)

        # step 4: create a dataframe with the flattened array
        variable_dataframe = pd.DataFrame(
            flat_variable_array,
            index=dates,  # Repeat dates for each row
            columns=cell_names,  # Repeat the created row names
        )

        print(f"Created {variable} number {str(i + 1)} / {str(len(variables))}")
        utils.save_pickle(
            data=variable_dataframe,
            file_name=f"{str(variable)}.pickle",
            directory=target_directory,
        )
        del variable_array
        del variable_dataframe
        del data_collection[variable]


def process_gldas_data(
    source_directory: str,
    target_directory: str,
) -> None:
    """
    Process GLDAS data from netCDF4 format to pandas dataframe format.

    Saves the data as a pickle file.

    :param source_directory: _description_
    :type source_directory: str
    :param target_directory: _description_
    :type target_directory: str
    """

    gldas = utils_data_classes.GLDASData()
    variables = gldas.variables
    files = gldas.get_file_names()

    dates = utils.get_date_range(files[0], files[-1])

    cell_names = ["Cell_" + str(i) for i in range(gldas.n_cells)]
    data_collection = {variable: [] for variable in variables}

    for i, variable in enumerate(variables):
        # step 1: open up the dataset, parse the array and flip it so that the orientation is correct
        for j, file in tqdm(enumerate(files)):
            dataset = nc.Dataset(os.path.join(source_directory, file))
            array = np.flip(
                np.squeeze(dataset.variables[variable][:], axis=0).data, axis=0
            )
            data_collection[variable].append(array)

        # step 2: concatenate arrays to make them two-dimensional
        variable_array = np.array(data_collection[variable])

        # step 3: flatten array by rows, then reshape it based on time (rows) and cells (columns)
        flat_variable_array = variable_array.reshape(-1, gldas.n_cells)

        # step 4: create a dataframe with the flattened array
        variable_dataframe = pd.DataFrame(
            flat_variable_array, index=dates[0 : len(files)], columns=cell_names[:]
        )

        print(f"Created {variable} number {str(i + 1)} / {str(len(variables))}")
        utils.save_pickle(
            data=variable_dataframe,
            file_name=f"{str(variable)}.pickle",
            directory=target_directory,
        )
        del variable_array
        del variable_dataframe
        del data_collection[variable]
