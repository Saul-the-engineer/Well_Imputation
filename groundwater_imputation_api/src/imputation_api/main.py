import sys

sys.path.append("..")
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
import utils
import utils_preprocess
import utils_spatial_interpolation
import utils_spatial_analysis

from config import Config
from utils_satellite_imputation import satellite_imputation
from utils_iterative_refinement import iterative_refinement


if __name__ == "__main__":
    # Load Shapefile
    path_shape = Config.shapefile_path
    aquifer_shape = utils.load_shapefile(path=path_shape)

    # Preprocess PDSI Data
    pdsi_source_directory = Config.pdsi_source_directory
    pdsi_target_directory = Config.pdsi_target_directory

    utils_preprocess.process_pdsi_data(
        source_directory=pdsi_source_directory,
        target_directory=pdsi_target_directory,
        date_start=Config.pdsi_preprocessing_start,
        date_end=Config.pdsi_preprocessing_end,
    )

    # Process the gldas netcdf files to obtain tabular data pickle file
    gldas_source_directory = Config.gldas_source_directory
    gldas_target_directory = Config.gldas_target_directory

    utils_preprocess.process_gldas_data(
        source_directory=gldas_source_directory,
        target_directory=gldas_target_directory,
    )

    # Parse pdsi data and save it
    directory_pdsi = Config.pdsi_target_directory

    pdsi: dict = utils.pull_relevant_data(
        shape=aquifer_shape,
        dataset_name=Config.pdsi_dataset_name,
        dataset_directory=directory_pdsi,
    )

    utils.save_pickle(
        data=pdsi,
        file_name=Config.pdsi_file_name,
        directory=Config.dataset_directory,
        protocol=3,
    )

    # Parse the GLDAS data and save it
    directory_gldas = Config.gldas_target_directory

    gldas: dict = utils.pull_relevant_data(
        shape=aquifer_shape,
        dataset_name=Config.gldas_dataset_name,
        dataset_directory=directory_gldas,
    )

    utils.save_pickle(
        data=gldas,
        file_name=Config.gldas_file_name,
        directory=Config.dataset_directory,
        protocol=3,
    )

    # Process well data from csv files
    well_locations = pd.read_csv(Config.well_locations_path)
    well_timeseries = pd.read_csv(Config.well_timeseries_path)

    data: dict = utils.transform_well_data(
        well_timeseries=well_timeseries,
        well_locations=well_locations,
        timeseries_name=Config.well_data_timeseries_name,
        locations_name=Config.well_data_locations_name,
    )

    utils.save_pickle(
        data=data,
        file_name=Config.well_data_file_name,
        directory=Config.dataset_directory,
        protocol=3,
    )

    satellite_imputation(
        aquifer_name=Config.aquifer_name,
        pdsi_pickle=Config.dataset_directory + "/" + Config.pdsi_file_name,
        gldas_pickle=Config.dataset_directory + "/" + Config.gldas_file_name,
        well_data_pickle=Config.dataset_directory + "/" + Config.well_data_file_name,
        output_file=Config.dataset_directory
        + "/"
        + Config.satellite_imputation_output_file,
        timeseries_name=Config.well_data_timeseries_name,
        locations_name=Config.well_data_locations_name,
        validation_split=0.3,
        folds=5,
    )

    iterative_refinement(
        aquifer_name=Config.aquifer_name,
        imputed_data_pickle=Config.dataset_directory
        + "/"
        + Config.satellite_imputation_output_file,
        output_file=Config.iterative_imputation_output_file,
        validation_split=0.3,
        folds=5,
        feature_threshold=0.60,
    )

    utils_spatial_interpolation.kriging_interpolation(
        data_pickle_path=Config.dataset_directory
        + "/"
        + Config.iterative_imputation_output_file,
        shape_file_path=Config.shapefile_path,
        n_x_cells=100,
        influence_distance=0.125,
        monthly_time_step=1,
        netcdf_filename=Config.raster_file_name,
        directory=Config.dataset_directory,
    )

    raster = nc.Dataset(
        Config.dataset_directory + "/" + Config.raster_file_name,
        "r",
    )

    spatial_analysis = utils_spatial_analysis.StorageChangeCalculator(
        units=Config.units,
        storage_coefficient=Config.storage_coefficient,
        anisotropic=Config.anisotropic,
    )
    storage_change = spatial_analysis.calulate_storage_curve(
        raster=raster,
        # date_range_filter=("1948-01-01", "1978-01-01"), # if you need to filter dates within of original time range
    )
