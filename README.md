<p align="center">
  <h2 align="center">Well Imputation: Using remote sensing information and iterative refinement to restore missing historical data</h2>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=Zrozd_gAAAAJ&hl=en"><strong>Saul Ramirez</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=3eyvLgkAAAAJ&hl=en"><strong>Gus Williams</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=S92pQn4AAAAJ&hl=en"><strong>Norm Jones</strong></a>
    <br>
    <b>Brigham Young University | &nbsp; NASA SERVIR</b>
</p>

  <table align="center">
    <tr>
    <td>
      <img src="https://user-images.githubusercontent.com/70539433/230738505-5caa500f-d46e-4d73-9016-58523ed5c663.png">
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/70539433/230738498-2a7e3dc7-1469-4c02-bd40-d376e5bd9e7c.png">
    </td>
    </tr>
  </table>

## 📢 News
* **[2023.12.04]** Update to project to increase reproducability!
* **[2023.03.22]** Release Improving Groundwater Imputation through Iterative Refinement Using Spatial and Temporal Correlations from In Situ Data with Machine Learning Paper.
* **[2022.11.01]** Release Groundwater level data imputation using machine learning and remote earth observations using inductive bias Paper.

## ⚒️ Installation
prerequisites: `python>=3.11`.  `CUDA>=11.2` recommended.

Install with `python`: 
`pip`:
```bash
pip3 install -r requirements.txt
```

## 🏃‍♂️ Getting Started
The overall structure of the project is as follows:

```bash
.
├── LICENSE
├── README.md
├── __pycache__
│   ├── data_classes.cpython-311.pyc
│   ├── utils.cpython-311.pyc
│   ├── utils_data_classes.cpython-311.pyc
│   └── utils_spatial.cpython-311.pyc
├── folder_tructure.txt
├── groundwater_imputation
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   ├── src
│   │   └── imputation_api
│   │       ├── __init__.py
│   │       ├── artifacts
│   │       │   ├── aquifer_data
│   │       │   ├── aquifer_figures
│   │       │   ├── aquifer_shapes
│   │       │   ├── dataset_outputs
│   │       │   └── figures
│   │       ├── config.py
│   │       ├── imputation_notebook.ipynb
│   │       ├── main.py
│   │       ├── routes
│   │       ├── sample_artifacts
│   │       │   ├── aquifer_data
│   │       │   ├── aquifer_shapes
│   │       │   └── dataset_outputs
│   │       ├── utils.py
│   │       ├── utils_data_classes.py
│   │       ├── utils_iterative_refinement.py
│   │       ├── utils_ml.py
│   │       ├── utils_plot.py
│   │       ├── utils_preprocess.py
│   │       ├── utils_satellite_imputation.py
│   │       ├── utils_spatial_analysis.py
│   │       └── utils_spatial_interpolation.py
│   └── tests
│       ├── artifacts
│       ├── fixtures
│       ├── functional_tests
│       └── unit_tests
└── version.txt
```
The project requires 4 datasets to function as intended:
* Aquifer Shapefile: Needs shapefile and metadata in WGS84. Placed in groundwater_imputation > imputation_api > artifacts > aquifer_shapes
* Well Data: This will be two .csv files, containing well locations, well measurements using a well id as a key. Placed in groundwater_imputation > imputation_api > artifacts > aquifer_data
* PDSI Extended Dataset: NetCDF of the [PDSI extended file](https://www.hydroshare.org/resource/145b386aa865459fb52a75e4230f6a14/). Placed locally on computer and pointed to within Config file.
* GLDAS Dataset: NASA GLDAS dataset [NASA](https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH025_3H.2.1/) or [Brigham Young University](https://drive.google.com/drive/u/0/folders/12XH-LUgK9-gBReIIAmxtIuHQ8tVTVjgA). Placed locally on computer and pointed to within Config file.

Sample artifact files are provided to dive into the project right away.

## 🙏 Acknowledgements
We would like to thank NASA SERVIR for funding this research project.

## 🎓 Citation
If you find this codebase useful for your research, please use the following entry.
```BibTeX
@article{ramirez2022groundwater,
  author = {Ramirez, Saul G. and Williams, Gustavious Paul and Jones, Norman L.},
  title = {Groundwater Level Data Imputation Using Machine Learning and Remote Earth Observations Using Inductive Bias},
  journal = {Remote Sensing},
  volume = {14},
  number = {21},
  year = {2022},
  pages = {5509},
  doi = {10.3390/rs14215509},
  url = {https://doi.org/10.3390/rs14215509}
}

@article{ramirez2022groundwater,
  author = {Ramirez, Saul G. and Williams, Gustavious Paul and Jones, Norman L.},
  title = {Groundwater Level Data Imputation Using Machine Learning and Remote Earth Observations Using Inductive Bias},
  journal = {Remote Sensing},
  volume = {14},
  number = {21},
  year = {2022},
  pages = {5509},
  doi = {10.3390/rs14215509},
  url = {https://doi.org/10.3390/rs14215509}
}
```

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
