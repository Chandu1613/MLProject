import os

os.chdir("../")

import pandas as pd
import zipfile
from logs import log_message

def load_data(file_path):
    log_message("Loading dataset...")

    unzip_path = "datasets/extracted_winequality"
    os.makedirs(unzip_path, exist_ok=True)  

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(unzip_path)

    extracted_folder = os.path.join(unzip_path, "winequality")

    if not os.path.exists(extracted_folder):
        raise FileNotFoundError(f"Expected folder 'winequality' not found in {unzip_path}")

    csv_files = [f for f in os.listdir(extracted_folder) if f.endswith(".csv")]

    if not csv_files:
        raise ValueError("No CSV file found inside the extracted 'winequality' folder.")

    csv_file_path = os.path.join(extracted_folder, csv_files[0])

    log_message(f"Reading CSV file: {csv_file_path}")

    df = pd.read_csv(csv_file_path)

    os.makedirs("artifacts", exist_ok=True)
    df.to_csv("artifacts/ingested_data.csv", index=False)

    return df
