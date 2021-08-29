import utils_spatial


data_root = './Datasets/' # Data Locations
figures_root = './Figures Spatial' # Location where figures are saved
netcdf_filename = 'well_data.nc' # Naming netcdf output
skip_month = 48 # Time interval of netcdf, published value 48 recomended 1.
num_cells = 100 # Script creates sqare grid num_cells x num_cells in size

# Initiate class creating data and figure folder
inter = utils_spatial.krigging_interpolation(data_root, figures_root)

# Load complete pickle file
well_data_dict = inter.read_pickle('Well_Data_Imputed', data_root)
well_data = well_data_dict['Data'].dropna() # Unpack Time series Drop NAs
x_coordinates = well_data_dict['Location']['Longitude'] # Unpack Longitude of wells
y_coordinates = well_data_dict['Location']['Latitude'] # Unpack Latitude of wells

# Load respective aquifer shape file
polygon = inter.Shape_Boundary('./Aquifer Shapes/Escalante_Beryl.shp')
# Line creates grid num_cells x num cells in the bounding box of the aquifer
# Creates boolean mask of cells located inside aquifer boundary to split interpolation map
grid_long, grid_lat = inter.create_grid_polygon(polygon, num_cells=num_cells)

# Extract every nth month of data
data_subset = inter.extract_dataframe_data(well_data, skip_month)

# This sets up a netcdf file to store each raster in.
file_nc, raster_data = inter.netcdf_setup(grid_long, grid_lat, data_subset.index, netcdf_filename)

# Loop through each date, create variogram for time step create krig map.
# Inserts the grid at the timestep within the netCDF.
for i, date in enumerate(data_subset.index):
    # Filter values associated with step
    values = data_subset.loc[data_subset.index[i]].values

    # fit the model variogram to the experimental variogram
    var_fitted = inter.fit_model_var(x_coordinates, y_coordinates, values, influence = 0.50)  # fit variogram
    # when kriging, you need a variogram. The subroutin has a function to plot
    # the variogram and the experimental. Variable 'influence' is the percentage
    # of the total aquifer length where wells are correlated. set 0.125 - 0.875

    krig_map = inter.krig_field(var_fitted, x_coordinates, y_coordinates, values, grid_long, grid_lat, date) # krig data
    # krig_map.field provides the 2D array of values
    # this function does all the spatial interpolation using the variogram from above.
    # Removes all data outside of boundaries of shapefile.

    # write data to netcdf file
    raster_data[i,:,:] = krig_map.field  # add kriged field to netcdf file at time_step

file_nc.close()

print('NetCDF Created.')