import utils
import logging
import math
import numpy as np
import pandas as pd
import netCDF4 as nc
from netCDF4 import Variable

logging.basicConfig(
    filename="storage_change.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
)


def create_spatial_interpolation_netcdf():
    pass


class StorageChangeCalculator:
    def __init__(self, units: str = "English", storage_coefficient: float = 0.2):
        """
        Initialize the StorageChangeCalculator.

        :param raster: The netCDF4 dataset containing raster data.
        :type raster: nc.Dataset
        :param units: The system of measurement for imputed data. Choose between "English" and "Metric".
        :type units: str, optional
        :param storage_coefficient: The storage coefficient of the aquifer.
        :type storage_coefficient: float, optional
        """

        self.units = units
        self.storage_coefficient = storage_coefficient
        self.anisotropic = "x"
        self.degrees_to_radians = math.pi / 180

        if units == "English":
            self.unit_coeff = 1
            self.area_coeff = 43560  # 1 acre = 43560 ft^2
            self.vol_coeff = 1  # 1 acre-ft = 1 acre * 1 ft
            self.conversion_factor = 5280  # 1 mile = 5280 ft
            self.conversion_factor2 = 1e6  # 1 million = 1e6
            self.radius_earth = (
                3958.8 * self.conversion_factor
            )  # radius of earth is 3958.8 miles
            self.calculation_unit = "feet"
            self.area_unit = "million acres"
            self.volume_unit = "million acre-feet"

        elif units == "Metric":
            self.unit_coeff = 0.3048  # 1 ft = 0.3048 m
            self.area_coeff = 1  # 1 km^2 = 1 km^2
            self.vol_coeff = 1000  # 1 km^3 = 1000 m^3
            self.conversion_factor = 1000  # 1 km = 1000 m
            self.conversion_factor2 = 1  # 1 = 1
            self.radius_earth = (
                6378.1 * self.conversion_factor
            )  # radius of earth is 6371 km
            self.calculation_unit = "meters"
            self.area_unit = "km^2"
            self.volume_unit = "km^3"

        else:
            logging.error(
                "Invalid units. Please choose between 'English' and 'Metric'."
            )
            return

        # Check anisotropy that the cells have the same resolution in one direction
        assert self.anisotropic in [
            "x",
            "y",
        ], "Invalid anisotropy. Please choose between 'x' and 'y'."
        logging.info("Cells have same resolution in {self.anisotropic} direction.")

        # Check units
        assert self.units in [
            "English",
            "Metric",
        ], "Invalid units. Please choose between 'English' and 'Metric'."
        logging.info("The units are set to %s.", self.units)

        # Check storage coefficient
        assert (
            self.storage_coefficient >= 0 and self.storage_coefficient <= 1
        ), "The storage coefficient must be between 0 and 1."
        logging.info(
            "The storage coefficient is set to %.2f.", self.storage_coefficient
        )

        return

    def calulate_storage_change_curve(
        self,
        raster: nc.Dataset,
        date_start: str,
        date_end: str,
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
        area = self.calculate_area(raster)
        logging.info("Area calculated.")
        logging.info(
            f"The area of the aquifer is: {round(area / self.conversion_factor2 / self.area_coeff)} {self.area_unit}"
        )

        logging.info("Calculating storage change...")
        storage_change = self.calculate_storage_change(raster, area)
        logging.info("Storage change calculated.")

        return storage_change

    def calulate_storage_change_curve_metric(
        self, raster: nc.Dataset, area: float, date_start: str, date_end: str
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
        self.check_time_dimension(raster)

        # Extract time index and validate date range
        datetime_index = self.extract_and_validate_time_index(
            raster, date_start, date_end
        )

        # Extract raster values and calculate differences
        raster_difference = self.calculate_raster_differences(raster)

        # Calculate drawdown volume
        storage_coefficient = self.storage_coefficient  # You should have this value
        drawdown_volume = self.calculate_drawdown_volume(
            raster_difference, storage_coefficient, area
        )

        # Create and adjust the storage change curve
        storage_change = self.create_adjust_storage_change_curve(
            drawdown_volume, datetime_index
        )

        return storage_change

    def check_time_dimension(self, raster: nc.Dataset):
        if "time" not in raster.dimensions:
            logging.error("No time dimension found.")
            raise ValueError("No time dimension found.")

    def extract_and_validate_time_index(
        self, raster: nc.Dataset, date_start: str, date_end: str
    ):
        datetime_units = raster["time"].units
        reference_date_str = datetime_units.split(" ")[-2]
        datetime_index = self.extract_netcdf_time_index(
            raster["time"], reference_date_str
        )

        assert (
            datetime_index[0] <= date_start <= datetime_index[-1]
        ), "The start date is not within the time range of the raster."
        assert (
            datetime_index[0] <= date_end <= datetime_index[-1]
        ), "The end date is not within the time range of the raster."
        assert date_start <= date_end, "The start date must be before the end date."

        return datetime_index

    def calculate_raster_differences(self, raster: nc.Dataset):
        raster_values = raster["tsvalue"][:]
        raster_difference = raster_values[0:-1] - raster_values[1:]
        raster_difference = np.insert(raster_difference, 0, 0, axis=0)
        return raster_difference

    def calculate_drawdown_volume(self, raster_difference, storage_coefficient, area):
        return np.nanmean(raster_difference * storage_coefficient * area, axis=(1, 2))

    def create_adjust_storage_change_curve(
        self, drawdown_volume, datetime_index, date_start, date_end
    ):
        storage_change = pd.Series(drawdown_volume, index=datetime_index)
        storage_change = storage_change[date_start:date_end]
        storage_change = storage_change - storage_change[0]
        return storage_change

    def calculate_area_longitude(
        self,
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
                (self.degrees_to_radians * x_resolution * num_cells)
                * (self.radius_earth) ** 2
                * abs((math.sin(mylatmin) - math.sin(mylatmax)))
            )

        return area

    def extract_netcdf_time_index(
        self, time_variable: Variable, datetime_units: str
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

        reference_date = dt.datetime.strptime(datetime_units, "%Y-%m-%d")
        cft_timestamps = nc.num2date(time_variable[:], reference_date)
        datetime_objects = [
            pd.Timestamp(dt.strftime("%Y-%m-%d %H:%M:%S")) for dt in cft_timestamps
        ]
        datetime_index = pd.DatetimeIndex(datetime_objects)
        return datetime_index
