import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.data_ingestion import load_data
from src.data_transformation import transform

if __name__ == "__main__":
    dataset_path = "datasets/winequality.zip"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found!")

    df = load_data(dataset_path)
    
    df = transform()
