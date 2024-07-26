# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:42:54 2024

@author: Malek
"""
import pandas as pd

def read_and_concatenate_csv(files):
    """Read and concatenate multiple CSV files into a single DataFrame."""
    return pd.concat(map(pd.read_csv, files), ignore_index=True)

def process_data(data, weather, capacity, holidays):
    """Process and merge data with weather, capacity, and holidays information."""
    
    # Convert date columns to datetime and extract additional features
    data.iloc[:, 8] = pd.to_datetime(data.iloc[:, 8], format="%Y%m%d")
    data["Week"] = data.iloc[:, 8].dt.dayofweek
    data['Hour'] = pd.to_datetime(data.iloc[:, 7]).dt.hour
    data.iloc[:, 8] = pd.to_datetime(data.iloc[:, 8], format="%Y%m%d").dt.strftime("%Y-%m-%d")

    # Select relevant columns from the data
    data = data.iloc[:, [4, 5, 7, 8, 10, 11, 12, 21, 22, 27, 28]]

    # Convert Weather date columns to datetime and extract additional features
    weather.iloc[:, 0] = pd.to_datetime(weather.iloc[:, 0], format="%Y-%m-%d").dt.strftime("%Y-%m-%d")
    weather['Hour'] = pd.to_datetime(weather.iloc[:, 1]).dt.hour

    # Ensure date columns are timezone-naive
    data['start_date'] = pd.to_datetime(data['start_date'], utc=True).dt.tz_localize(None)
    weather['date'] = pd.to_datetime(weather['date']).dt.tz_localize(None)

    # Merge datasets
    data = pd.merge(data, weather, left_on=['start_date', 'Hour'], right_on=['date', 'Hour'])
    data = pd.merge(data, capacity, left_on=['vehicle_id'], right_on=['Vessels'])
    data = pd.merge(data, holidays, left_on=['start_date'], right_on=['Date'])

    # Select relevant columns from the merged data
    data = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 29]]

    return data

def save_data_to_csv(data, filename):
    """Save the processed data to a CSV file and print the number of observations."""
    data.to_csv(filename, index=False)
    print(f"Number of observations in {filename}: {data.shape[0]}")

def main():
    # Configuration
    csv_files = ['2022-Data.csv', '2023-Data.csv']
    weather_file = "Weather.xlsx"
    capacity_file = "Capacity.xlsx"
    holidays_file = "Holidays2.xlsx"
    output_file = 'merged1.csv'

    try:
        # Read CSV and Excel files
        data = read_and_concatenate_csv(csv_files)
        weather = pd.read_excel(weather_file)
        capacity = pd.read_excel(capacity_file)
        holidays = pd.read_excel(holidays_file)

        # Process and merge data
        processed_data = process_data(data, weather, capacity, holidays)

        # Save processed data to CSV
        save_data_to_csv(processed_data, output_file)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
