from dataclasses import dataclass


@dataclass
class Config:
    # Shapefile paths
    shapefile_path: str = "groundwater_imputation/src/imputation_api/artifacts/aquifer_shapes/Beryl_Enterprise.shp"
    well_locations_path: str = "groundwater_imputation/src/imputation_api/artifacts/aquifer_data/EscalanteBerylLocation.csv"
    well_timeseries_path: str = "groundwater_imputation/src/imputation_api/artifacts/aquifer_data/EscalanteBerylTimeseries.csv"

    # Directories
    pdsi_source_directory: str = (
        "groundwater_imputation/src/imputation_api/artifacts/pdsi_dataset"
    )
    pdsi_target_directory: str = (
        "groundwater_imputation/src/imputation_api/artifacts/pdsi_tabular"
    )
    gldas_source_directory: str = (
        "groundwater_imputation/src/imputation_api/artifacts/gldas_dataset"
    )
    gldas_target_directory: str = (
        "groundwater_imputation/src/imputation_api/artifacts/gldas_tabular"
    )
    dataset_directory: str = (
        "groundwater_imputation/src/imputation_api/artifacts/dataset_outputs"
    )

    # Date-related configurations
    pdsi_preprocessing_start: str = "01/01/1850"
    pdsi_preprocessing_end: str = "12/31/2020"
    well_processing_start: str = "1/1/1948"
    well_processing_end: str = "1/1/2020"

    # Preprocessing hyperparameters
    well_padding: int = 90  # days
    well_gap_size: int = 365  # days
    well_min_observations: int = 50  # months
    well_max_std: int = 3  # std

    # Imputation hyperparameters
    validation_split: float = 0.3
    folds: int = 5
    feature_threshold: float = 0.60

    # Dataset names and file names
    pdsi_dataset_name: str = "PDSI"
    pdsi_file_name: str = "pdsi_data.pickle"
    gldas_dataset_name: str = "GLDAS"
    gldas_file_name: str = "gldas_data.pickle"
    well_data_file_name: str = "beryl_enterprise_data.pickle"
    well_data_timeseries_name: str = "timeseries"
    well_data_locations_name: str = "locations"

    # Aquifer information
    aquifer_name: str = "Beryl Enterprise"

    # Output files
    satellite_imputation_output_file: str = (
        "beryl_enterpris_imputation_satellite.pickle"
    )
    iterative_imputation_output_file: str = "beryl_enterprise_iterative.pickle"

    # Spatial constants
    units: str = "English"
    storage_coefficient: float = 0.2
    anisotropic: str = "x"
    number_of_x_cells: int = 100
    influence_distance: float = 0.125
    monthly_time_step: int = 1
    date_filter_start: str = "1948-01-01"
    date_filter_end: str = "1978-01-01"

    # Spatial analysis
    raster_file_name: str = "beryl_enterprise_spatial_analysis.nc"
