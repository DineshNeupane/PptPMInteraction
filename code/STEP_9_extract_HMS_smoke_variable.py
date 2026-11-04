
import geopandas as gpd
import pandas as pd
import zipfile
import os
from shapely.geometry import Point
from datetime import datetime, timedelta
import shutil

# Input CSV file with Latitude and Longitude points
input_csv = r"\data\coordinates\unique_coordinates_with_years_and_id.csv"

# Folder containing all shapefile ZIPs
shapefile_folder = r"\data\HMSSmoke"

# Output CSV file to store all results
output_csv = r"\data\smoke\extracted_smoke_data_for_10_years_pm10_coords.csv"

skipped_files = 0


# Function to convert YYYYDDD HHMM format to datetime
def convert_julian_datetime(julian_str):
    try:
        if isinstance(julian_str, str) and " " in julian_str:
            date_part, time_part = julian_str.split(" ")

            # Extract year and day of the year
            year = int(date_part[:4])
            day_of_year = int(date_part[4:])

            # Convert Julian day to standard date
            date_obj = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)

            # Extract hour and minute
            hour = int(time_part[:2])
            minute = int(time_part[2:])

            # Combine into a full datetime object
            final_datetime = datetime(date_obj.year, date_obj.month, date_obj.day, hour, minute, 0)

            return final_datetime.strftime("%Y-%m-%d %H:%M:%S")  # Convert to string format
        else:
            return "Invalid Format"
    except Exception:
        return "Conversion Error"


# Load input points from CSV
input_points = pd.read_csv(input_csv)

# Get a list of all ZIP files in the folder
shapefile_zips = [os.path.join(shapefile_folder, f) for f in os.listdir(shapefile_folder) if f.endswith(".zip")]

# Initialize an empty list to store results
all_results = []

# Iterate over each shapefile ZIP
for zip_path in shapefile_zips:
    print(f"Processing {zip_path}...")

    # Extract the ZIP file to a temporary directory
    extract_dir = zip_path.replace(".zip", "")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)

    # Find the shapefile (.shp) in the extracted directory
    shapefile_path = None
    for file in os.listdir(extract_dir):
        if file.endswith(".shp"):
            shapefile_path = os.path.join(extract_dir, file)
            break

    if shapefile_path:
       try:
           # Load the shapefile into a GeoDataFrame
           gdf = gpd.read_file(shapefile_path)

           # Ensure the CRS is WGS84
           if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
               gdf = gdf.to_crs(epsg=4326)

           # Iterate over each point
           for index, row in input_points.iterrows():
               latitude = row["Latitude"]
               longitude = row["Longitude"]
               point = Point(longitude, latitude)

               # Find intersecting polygons
               intersects = gdf[gdf.geometry.intersects(point)]

               # Append intersecting polygon data to results
               for idx, poly in intersects.iterrows():
                   start_date = convert_julian_datetime(str(poly.get("Start", "N/A")))
                   end_date = convert_julian_datetime(str(poly.get("End", "N/A")))
                   density = poly.get("Density", "N/A")

                   all_results.append({
                       "Latitude": latitude,
                       "Longitude": longitude,
                       "Shapefile": os.path.basename(zip_path),  # Store the shapefile name
                       "Start": start_date,
                       "End": end_date,
                       "Density": density
                   })
       except Exception as e:
           print(f"Skipping {shapefile_path} due to error: {e}")
           skipped_files += 1
           shutil.rmtree(extract_dir)  # clean up
           continue  # skip to the next file
    else:
        print(f"No shapefile found in {zip_path}.")

    # Clean up extracted files after processing
    shutil.rmtree(extract_dir)

# Create a DataFrame from all results
if all_results:
    output_df = pd.DataFrame(all_results)

    # Save to CSV
    output_df.to_csv(output_csv, index=False)
    print(f"CSV file '{output_csv}' created successfully with {len(output_df)} records.")
else:
    print("No intersecting polygons found in any shapefiles.")
