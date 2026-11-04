import os
import re
from datetime import datetime
import pandas as pd
import xarray as xr
import warnings
import numpy as np
import time
import logging
import ast
from joblib import Parallel, delayed

warnings.simplefilter("ignore")

log_filename = r"\data\ALL_PM_10_DATA\logs\process_2023.log"
folder_path = r"\data\nldas_subset_data"
file_extension = ".nc4"
output_csv_path = r"\data\ALL_PM_10_DATA\processed_data\processed_data_multiprocessed_2023.csv"
coordinates_file = r'\data\coordinates\yearly_coordinates_pm10\unique_coordinates_with_years_and_id.csv'
use_test_mode = False

band_names = [
    "APCP",   # Total precipitation
    "CAPE",
    "CONVfrac",
    "DLWRF",
    "DSWRF",
    "PEVAP",
    "PRES",
    "SPFH",
    "TMP",
    "UGRD",
    "VGRD"
]

# ----------------- LOGGING -----------------
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# ----------------- COORDINATE DATA -----------------
coord_df = pd.read_csv(coordinates_file)
coord_df['Years_Present'] = coord_df['Years_Present'].apply(ast.literal_eval)

# ----------------- HELPERS -----------------
def log_time(message, start_time):
    elapsed = time.perf_counter() - start_time
    logging.info(f"{message} took {elapsed:.2f} seconds")
    return time.perf_counter()

# -----------------FILENAME PARSER -----------------
def extract_datetime_from_filename(filename):
    # Pattern for: NLDAS_FORA0125_H.A20230101.0000.002.grb.SUB.nc4
    pattern = r"NLDAS_\w+_H\.A(\d{8})\.(\d{4})\."
    match = re.search(pattern, filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        return datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
    else:
        logging.warning(f"Filename {filename} does not match expected pattern.")
        return None

def find_nearest(array, values):
    idx = np.abs(array - values[:, None]).argmin(axis=1)
    return idx

def process_file(file_name):
    file_path = os.path.join(folder_path, file_name)
    local_data = []
    try:
        t0 = time.perf_counter()
        data = xr.open_dataset(file_path, engine='netcdf4')
        t0 = log_time(f"Opened file {file_name}", t0)

        datetime_obj = extract_datetime_from_filename(file_name)
        if datetime_obj is None:
            return []

        year = datetime_obj.year
        coords_this_year = coord_df[coord_df['Years_Present'].apply(lambda y: year in y)].reset_index(drop=True)
        t0 = log_time("Filtered coordinates", t0)

        if coords_this_year.empty:
            logging.info(f"No coordinates for year {year} in {file_name}, skipping.")
            return []

        lat_var = data['lat'].values
        lon_var = data['lon'].values

        lat_idx = find_nearest(lat_var, coords_this_year['Latitude'].values)
        lon_idx = find_nearest(lon_var, coords_this_year['Longitude'].values)
        t0 = log_time("Computed nearest lat/lon indices", t0)

        for i, (lat_i, lon_i) in enumerate(zip(lat_idx, lon_idx)):
            selected = data.isel(lat=lat_i, lon=lon_i)

            # pull all bands from this cell
            values = selected[band_names].to_array().values.flatten()

            band_data = dict(zip(band_names, values))
            band_data["Latitude"] = coords_this_year.loc[i, "Latitude"]
            band_data["Longitude"] = coords_this_year.loc[i, "Longitude"]
            band_data["UID"] = coords_this_year.loc[i, "UID"]
            band_data["date_time"] = datetime_obj

            if not any(pd.isna(v) for v in band_data.values()):
                local_data.append(band_data)

        t0 = log_time("Extracted data for coordinates", t0)
        logging.info(f"Finished processing file: {file_name}")

    except Exception as e:
        logging.error(f"Error processing file {file_name}: {e}")
        return []

    return local_data

def chunks(lst, batch_sz):
    for i in range(0, len(lst), batch_sz):
        yield lst[i:i + batch_sz]

# ----------------- MAIN -----------------
if __name__ == "__main__":
    start_time = time.time()

    all_files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
    if use_test_mode:
        all_files = all_files[:5]

    logging.info(f"Found {len(all_files)} files to process")

    batch_size = 200
    all_results = []

    for batch_files in chunks(all_files, batch_size):
        logging.info(f"Processing batch of {len(batch_files)} files")

        batch_results = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(process_file)(f) for f in batch_files
        )

        # flatten
        for sub in batch_results:
            all_results.extend(sub)

    output_df = pd.DataFrame(all_results)
    output_df.to_csv(output_csv_path, index=False)

    logging.info(f"Processed {len(output_df)} valid records.")
    logging.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
