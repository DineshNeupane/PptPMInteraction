import os
import requests
from datetime import datetime, timedelta
import time

# Output directory to store the downloaded shapefiles
output_dir = r"../data/HMSSmoke"
os.makedirs(output_dir, exist_ok=True)

# Define the start and end dates across 10 years
start_date = datetime(2014, 1, 1)
end_date = datetime(2023, 12, 31)

# Iterate through each day in the date range
current_date = start_date
while current_date <= end_date:
    year = current_date.strftime("%Y")
    month = current_date.strftime("%m")
    day = current_date.strftime("%d")

    # Dynamically construct the base URL
    base_url = f"https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile/{year}/{month}/"
    filename = f"hms_smoke{year}{month}{day}.zip"
    file_url = f"{base_url}{filename}"

    try:
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(os.path.join(output_dir, filename), "wb") as file:
                file.write(response.content)
            print(f"Downloaded: {filename}")
        else:
            print(f"File not found for date: {year}-{month}-{day}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

    current_date += timedelta(days=1)
    time.sleep(0.1)  # Sleep briefly to avoid overwhelming the server
