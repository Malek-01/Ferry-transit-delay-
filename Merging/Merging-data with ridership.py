# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:42:54 2024

@author: Malek
"""
import pandas as pd

def read_csv_file(file_path):
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(file_path)

def read_excel_file(file_path):
    """Read an Excel file into a DataFrame."""
    return pd.read_excel(file_path)

def process_data(data, ridership):
    """Process and merge data with ridership information."""
    
    # Convert date column to datetime and extract year and month
    data['start_date'] = pd.to_datetime(data['start_date'])
    data['year'] = data['start_date'].dt.year
    data['month'] = data['start_date'].dt.month

    # Adjust route_id format in data
    data['route_id'] = data['route_id'].str.extract(r'(\d+-F\d+)')
    data['route_id'] = data['route_id'].str.replace('9-', '')

    # Adjust Line format in ridership
    ridership['Line'] = ridership['Line'].str.replace(' ', '')

    # Merge datasets
    data = pd.merge(data, ridership, left_on=['route_id', 'year', 'month'], right_on=['Line', 'Year', 'Month'])

    # Drop unnecessary columns
    data = data.drop(columns=['Line', 'year', 'month', 'Year', 'Month'])

    return data

def save_data_to_csv(data, filename):
    """Save the processed data to a CSV file and print the number of observations."""
    data.to_csv(filename, index=False)
    print(f"Number of observations in {filename}: {data.shape[0]}")

def main():
    # File paths
    data_file = 'Merged1.csv'
    ridership_file = 'Ridership3.xlsx'
    output_file = 'Merged-final.csv'

    try:
        # Read CSV and Excel files
        data = read_csv_file(data_file)
        ridership = read_excel_file(ridership_file)

        # Process and merge data
        processed_data = process_data(data, ridership)

        # Save processed data to CSV
        save_data_to_csv(processed_data, output_file)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
