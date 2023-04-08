# Well_Imputation
## Background
The purpose of the software used in this project is to provide a solution for the imputation of missing data in sparse time series datasets, with a particular focus on groundwater management. Groundwater-level records can be sparse, especially in developing areas, making it challenging to accurately characterize aquifer-storage change over time. This process typically begins with an analysis of historical water levels at observation wells. However, missing data can be a significant issue, leading to incomplete and potentially unreliable analyses.

To address this challenge, the project employs the methods of imputation based on [inductive bias](https://www.mdpi.com/2072-4292/14/21/5509) and [iterative refinement model (IRM)](https://www.mdpi.com/2073-4441/15/6/1236) machine learning framework published in Remote Sensing and Water respectively. This approach works on any aquifer dataset where each well has a complete record that can be a mixture of measured and input values. 

This process is applied in two steps: Inductive bias imputation and Iterative refinement imputation.

Inductive bias imputation is based on the idea that groundwater is correlated (loosely) to meteorological parameters such as precipitation and temperature. So we build a basic imputation model for each well in an aquifer based on remote sensing data from the [Palmer Drought Severity Index](https://www.hydroshare.org/resource/145b386aa865459fb52a75e4230f6a14/) and [Global Land Data Assimilation System](https://disc.gsfc.nasa.gov/). We use indcutive bias to generate an initial prediction of what values during the missing time periods could be. Inductive bias helps to improve the accuracy of the results by incorporating prior knowledge and assumptions about the underlying physical processes. This allows the model to make more informed decisions and generate more reliable predictions, even when limited data is available. Generally this is done based on the data centric prior, but other methods could be used.

This first approach generally creates annomalies and patterns that don't match with observed data from nearby wells at the same time period. Therefore, once we implement inductive bias imputation, we can apply iterative refinement imputation.

The IRM method involves selecting a small set of imputed time series datasets from the wells correlated to the target well, developing a model for the target well using the selected data, and running the model to generate a complete time series. The results of every model are updated synchronously at the end of each iteration, and the process is repeated for a selected number of iterations. The use of a Hampel filter helps to smooth synthetic data spikes or model predictions that are unrealistic for groundwater data, while the selection of wells based on linear correlation and spatial distance aids in developing a more accurate model.
![image](https://user-images.githubusercontent.com/70539433/230738483-b8502492-5bfa-423f-87b2-701e5702d00e.png)

## Sample Results
We share some sample results from the Beryl-Enterprise Aquifer in Utah. An explanation of the results in given in: [IRM](https://www.mdpi.com/2073-4441/15/6/1236)
![image](https://user-images.githubusercontent.com/70539433/230738498-2a7e3dc7-1469-4c02-bd40-d376e5bd9e7c.png)
![image](https://user-images.githubusercontent.com/70539433/230738502-d9184c15-5051-438a-bd05-dcb434d6de9e.png)
![image](https://user-images.githubusercontent.com/70539433/230738505-5caa500f-d46e-4d73-9016-58523ed5c663.png)
![image](https://user-images.githubusercontent.com/70539433/230738515-8af95061-1af0-4d4d-bd17-2cc02e207878.png)

## Environment (Python)
```
python==3.8.10
numpy==1.22.3
pandas==1.3.5
h5py==3.7.0
tables==3.8.0
scipy==1.6.2
scikit-learn==1.12
Fiona==1.8.13.post1
rasterio==1.3.6
geopandas==0.9.0
shapely==1.8.4
gstools==1.4.0
grids # 0.15
netCDF4 # 1.5.7
tensorflow # 2.5.0
pickle5
```

## Data Requirements
To get started you will need the a csv of timeseries data, a csv of well locations, a shape file of the respective aquifer, the Palmer Drought Severity index, and the GLDAS Tabular dataset. Palmer Drought Severity index, and the GLDAS Tabular dataset are hosted in [Google drive](https://drive.google.com/drive/u/0/folders/1hSN6gkp9zmFYUwMOdDIj8pa-KqBjN8JW) with regular updates by [Brigham Young University](http://hydroinf.groups.et.byu.net/servir-wa/). Check out the group page to learn more about our mission and future work.

The tabular GLDAS is parsing each individual GLDAS variable from every month and saving it as it's own netCDF. This is done because GLDAS is distributed as a monthly netCDF with each variable as a layer in the file. In the future, this may be updated to a SQLite database hosted on google drive as the files are begining to get very large. The files are hosted on Google Drive to help with badwidth issues in Western Africa where the sponsers of this project are located. Being able to mount data to a personal google drive, rather than downloaded the ~24 GB dataset (as of April 2023) seemed more feasable as our partners had bandwidth problems.

## Implementation
Data_Imputation_Compiled.ipynb gives a comprehensive overview of how to apply this framework.
A sample dataset is included.
