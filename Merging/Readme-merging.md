# Ferry Transit Delay Data Processing

This repository contains scripts for processing and merging ferry transit delay data with additional datasets such as weather, capacity, holidays, and ridership information.

## Files

- `Merging-data.py`: Script for reading, processing, and merging multiple CSV and Excel files into a single dataset.
- `Merging-data with ridership.py`: Script for merging the generated file `Merged1.csv` with ridership information from `Ridership3.xlsx`.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file.

## Requirements

- Python 3.6 or higher
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Malek-01/Ferry-transit-delay-.git
    cd Ferry-transit-delay-/Merging
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Merging Multiple CSV and Excel Files

The `Merging-data.py` script reads multiple CSV files and merges them with weather, capacity, and holidays data from Excel files.

1. Ensure you have the necessary data files:
    - `2022-Data.csv`
    - `2023-Data.csv`
    - `Weather.xlsx`
    - `Capacity.xlsx`
    - `Holidays2.xlsx`

2. Run the script:
    ```sh
    python Merging-data.py
    ```

The processed data will be saved to `merged1.csv`.

### Merging with Ridership Information

The `Merging-data with ridership.py` script reads `Merged1.csv` and merges it with ridership to generate the file that would be processed.