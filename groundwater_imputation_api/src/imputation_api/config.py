from dataclasses import dataclass


@dataclass
class Config:
    # Shapefile paths
    shapefile_path: str = "/home/saul/workspace/groundwater_well_imputation/groundwater_imputation_api/src/imputation_api/artifacts/aquifer_shapes/Beryl_Enterprise.shp"
    well_locations_path: str = "/home/saul/workspace/groundwater_well_imputation/groundwater_imputation_api/src/imputation_api/artifacts/aquifer_data/EscalanteBerylLocation.csv"
    well_timeseries_path: str = "/home/saul/workspace/groundwater_well_imputation/groundwater_imputation_api/src/imputation_api/artifacts/aquifer_data/EscalanteBerylTimeseries.csv"

    # Directories
    pdsi_source_directory: str = r"C:\Users\saulg\Desktop\Remote_Data\pdsi"
    pdsi_target_directory: str = r"C:\Users\saulg\Desktop\Remote_Data\pdsi_tabular"
    gldas_source_directory: str = r"C:\Users\saulg\Desktop\Remote_Data\GLDAS"
    gldas_target_directory: str = r"C:\Users\saulg\Desktop\Remote_Data\gldas_tabular"
    dataset_directory: str = "/home/saul/workspace/Well_Imputation/groundwater_imputation_api/src/imputation_api/artifacts/dataset_outputs"

    # Date-related configurations
    pdsi_preprocessing_start: str = "01/01/1850"
    pdsi_preprocessing_end: str = "12/31/2020"

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

    # Constants
    units: str = "English"
    storage_coefficient: float = 0.2
    anisotropic: str = "x"

    # Spatial analysis
    raster_file_name: str = "beryl_enterprise_spatial_analysis.nc"
