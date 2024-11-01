import zipfile
import os
from pathlib import Path
import pandas as pd

def unzip_kraken_history():
    """
    Unzips Kraken trading history data files from the Kraken_Trading_History.zip file.
    """
    # Define paths
    zip_file = Path(r"C:\Users\harmo\OneDrive\Desktop\algo_0\StrategyTest\Kraken_Trading_History.zip")
    extract_dir = zip_file.parent / "Kraken_Trading_History"
    
    # Create extraction directory if it doesn't exist
    if not extract_dir.exists():
        extract_dir.mkdir()
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            print(f"Successfully extracted {zip_file.name} to {extract_dir}")
    except zipfile.BadZipFile:
        print(f"Error: {zip_file.name} is not a valid zip file")
    except Exception as e:
        print(f"Error extracting {zip_file.name}: {str(e)}")

def load_crypto_data():
    """
    Loads all CSV files from Kraken_Trading_History directory into a single DataFrame.
    Returns a pandas DataFrame with all trading history data.
    """
    data_dir = Path(r"C:\Users\harmo\OneDrive\Desktop\algo_0\StrategyTest\Kraken_Trading_History")
    all_data = []
    
    # Debug: Print the directory we're looking in
    print(f"Looking for files in: {data_dir}")
    
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return pd.DataFrame()
    
    # Look for CSV files in the directory
    for csv_file in data_dir.glob("*.csv"):
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            all_data.append(df)
            print(f"Successfully loaded {csv_file}")
        except Exception as e:
            print(f"Error loading {csv_file}: {str(e)}")
    
    # Combine all DataFrames
    if all_data:
        crypto_data = pd.concat(all_data, ignore_index=True)
        print(f"Total rows loaded: {len(crypto_data)}")
        return crypto_data
    else:
        print("No data files found")
        return pd.DataFrame()

# Execute both functions
unzip_kraken_history()
crypto_data = load_crypto_data()
