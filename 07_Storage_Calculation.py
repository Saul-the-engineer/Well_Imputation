# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 00:48:16 2023

@author: saulg
"""

import netCDF4 as nc
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
import datetime

raster_file = "well_data_iter_1.nc"
#raster_file = "well_data_iter_2.nc"
root = r"C:\Users\saulg\OneDrive\Research\Well Imputation\Master Code\Datasets"


units = "English" #@param ["English", "Metric"]
if units == "Metric": 
  unit_coeff, area_coeff, vol_coeff, area_lab, vol_lab = 0.3048, 1, 1000, 'km^2', 'km^3'
else: 
  unit_coeff, area_coeff, vol_coeff, area_lab, vol_lab = 1, 43560, 1, 'million acres', 'million ac-ft'
#@markdown *Specify the storage coefficient of the aquifer*
storage_coefficient = 0.2 #@param {type:"number"}

imputed_raster = nc.Dataset(root + "/" +raster_file)

# Calculate the area of the aquifer
yRes = abs(round(imputed_raster['lat'][0] - imputed_raster['lat'][1], 7)) # this assumes all cells will be the same size in one dimension (all cells will have same x-component)
xRes = abs(round(imputed_raster['lon'][0] - imputed_raster['lon'][1], 7))
area = 0
# Loop through each y row
for y in range(imputed_raster.dimensions['lat'].size):
  # Define the upper and lower bounds of the row
  mylatmax = math.radians(imputed_raster['lat'][y] + (yRes/2))
  mylatmin = math.radians(imputed_raster['lat'][y] - (yRes/2))

  # Count how many cells in each row are in aquifer (i.e. and, therefore, not nan)
  xCount = 0
  for x in range(imputed_raster.dimensions['lon'].size):
    if not math.isnan(imputed_raster['tsvalue'][0, y, x]):
      xCount += 1
  
  # Area calculated based on the equation found here: https://www.pmel.noaa.gov/maillists/tmap/ferret_users/fu_2004/msg00023.html
  #     (pi/180) * R^2 * |lon1-lon2| * |sin(lat1)-sin(lat2)| 
  #     radius is 3958.8 mi
  area += (3958.8 * 5280 * unit_coeff)**2 * math.radians(xRes * xCount) * abs((math.sin(mylatmin) - math.sin(mylatmax)))
print("The area of the aquifer is %.2f %s.\n" %(area / 10**6 / area_coeff, area_lab))

# Calculate total drawdown volume at each time step
drawdown_grid = np.zeros((imputed_raster.dimensions['time'].size, imputed_raster.dimensions['lat'].size, imputed_raster.dimensions['lon'].size))
drawdown_volume = np.zeros(imputed_raster.dimensions['time'].size)
for t in range(imputed_raster.dimensions['time'].size):
  drawdown_grid[t, :, :] = imputed_raster['tsvalue'][t, :, :] - imputed_raster['tsvalue'][0, :, :] # Calculate drawdown at time t by subtracting original WTE at time 0
  drawdown_volume[t] = np.nanmean(drawdown_grid[t, :, :] * storage_coefficient * area ) # Average drawdown across entire aquifer * storage_coefficient * area of aquifer

# Plot storage depletion curve
x_data = pd.Series(imputed_raster['time'][:], dtype=int).apply(lambda x: datetime.datetime.fromordinal(x)) # Convert from days since 0001-01-01 00:00:00
y_data = drawdown_volume / 10**6 / area_coeff / vol_coeff
plt.plot(x_data, y_data)
plt.xlabel("Year")
plt.ylabel("Drawdown Volume (%s)" %(vol_lab))
plt.title(f"Storage Depletion Curve: S = {storage_coefficient}")
plt.show()

ztest = pd.DataFrame(y_data, index = x_data)
ztest0 = ztest[(ztest.index == "1977-10-02 00:00:00")].values

from sklearn.linear_model import LinearRegression
yreg = ztest
#yreg = ztest[(ztest.index >= "2000-01-02 00:00:00") & (ztest.index < "2013-01-02 00:00:00")]
#yreg = ztest[(ztest.index >= "1948-01-02 00:00:00") & (ztest.index < "2013-01-02 00:00:00")]
xreg = np.array([*range(len(yreg))]).reshape(-1,1)

reg = LinearRegression().fit(xreg, yreg.values)
reg.coef_ * 12

"""
ztest20 = ztest[(ztest.index == "2009-10-02 00:00:00")]
ztest30 = ztest[(ztest.index == "2010-10-02 00:00:00")]
ztest40 = (ztest30.values - ztest20.values)

ztest21 = ztest[(ztest.index == "2010-10-02 00:00:00")]
ztest31 = ztest[(ztest.index == "2011-10-02 00:00:00")]
ztest41 = (ztest31.values - ztest21.values)

ztest22 = ztest[(ztest.index == "2011-10-02 00:00:00")]
ztest32 = ztest[(ztest.index == "2012-10-02 00:00:00")]
ztest42 = (ztest32.values - ztest22.values)

ztest23 = ztest[(ztest.index == "2012-10-02 00:00:00")]
ztest33 = ztest[(ztest.index == "2013-10-02 00:00:00")]
ztest43 = (ztest33.values - ztest23.values)

zout = (ztest40 + ztest41 + ztest42 + ztest43)/4*1000000
"""
