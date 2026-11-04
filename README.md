## PM Removal by Precipitation Across the United States

This repository contains the data-processing, modeling, analysis, and visualization code used for the manuscript:

**“PM2.5 and PM10 Removal by Precipitation Across the United States Using Double Machine Learning: General and Wildfire Smoke Conditions”**

The study uses hourly EPA AQS PM2.5 and PM10 observations from 2014–2023, NLDAS-2 meteorology, and NOAA HMS wildfire-smoke classifications.

### Repository Structure

```text
PptPMInteraction/
├── code/
├── config/
│   ├── config.json
│   └── requirements.txt
├── data/
│   ├── aqs_data_10_years/
│   ├── coordinates/
│   ├── HMSsmoke/
│   ├── nldas_subset_data/
│   ├── smoke/
│   └── us_states_shapefile/
├── figures/
├── meta/
├── reports/
├── results/
│   ├── pm10/
│   └── pm25/
└── README.md
```

### Data Sources

- **EPA AQS**: hourly PM2.5 and PM10 observations
- **NLDAS-2**: hourly meteorological variables
- **NOAA HMS**: wildfire-smoke polygons and smoke-density categories

The study uses the smoke categories NS, Light, Medium, and Heavy. 

### Analysis Workflow

Run the files in the `code/` folder in numerical order.

#### Data Processing

```text
STEP_1_initial_processing_pipeline.ipynb
STEP_2_coordinate_encoding.ipynb
STEP_3_metrological_variable_extraction.py
STEP_4_assign_uid_and_data_selection.ipynb
STEP_5_combine_all_time_series_metro_data.ipynb
STEP_6_combine_PM_and_metro_data.ipynb
STEP_7_PM_metro_data_cleanup.ipynb
STEP_8_download_HMS_smoke_polygon_2014_2023.py
STEP_9_extract_HMS_smoke_variable.py
STEP_10_HMS_Smoke_Hourly_designation.ipynb
STEP_11_combine_pmmetro_w_smoke.ipynb
```

##### Modeling

```text
STEP_11_model_pm25.ipynb
STEP_12_model_PM10.py
```

The models use precipitation as the treatment, PM removal efficiency as the outcome, and meteorological and smoke variables as covariates in a Double Machine Learning analysis. 

### Post-Model Analysis

```text
STEP_13_post_model_analys_PM25.ipynb
STEP_14_post_model_analys_PM10.ipynb
STEP_15_removal_by_precip_THETA.ipynb
STEP_15_removal_by_precipitation_TOTAL.ipynb
STEP_16_removal_by_baseline_pm_THETA.ipynb
STEP_17_removal_by_baseline_pm_TOTAL_UNADJUSTED.ipynb
STEP_18_similarity_of_pm_distribution_smoke_class.ipynb
STEP_19_removal_by_smoke_density.ipynb
STEP_20_curve_fitting_for_a_and_b.ipynb
```

Smoke-group comparisons use nearest-neighbor matching of baseline PM distributions before estimating removal rates. 

#### Environment Setup

Install the required packages from the repository root:

```bash
pip install -r config/requirements.txt
```

#### Main Output Locations

```text
figures/        Final plots and manuscript figures
results/pm25/   PM2.5 model and analysis outputs
results/pm10/   PM10 model and analysis outputs
reports/        Optional summary outputs
```

#### Reproducing the Analysis (bring it to top)

1. Download the EPA AQS, NLDAS-2, and NOAA HMS source data. (mention the list of the files that does it---run the code to download the files)
2. Place the data in the corresponding folders under `data/`.
3. Check file paths in `config/config.json` and the scripts.
4. Install dependencies.
5. Run the processing files in numerical order.
6. Run the PM2.5 and PM10 models.
7. Run the post-model notebooks.
8. Generate the final figures and result tables.

#### Data Availability

- EPA AQS: <https://aqs.epa.gov/aqsweb/airdata/download_files.html>
- NOAA HMS: <https://www.ospo.noaa.gov/products/land/hms.html#data>
- NLDAS-2: <https://ldas.gsfc.nasa.gov/nldas/nldas-2-model-data>

These are the public data sources listed in the manuscript.

## Authors

Dinesh Neupane  
Viviana Maggioni  
Lucas Henneman  

Department of Civil, Environmental and Infrastructure Engineering  
George Mason University

## Citation

The final article citation and DOI will be added after publication.
