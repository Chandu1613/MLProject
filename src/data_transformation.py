import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logs import log_message

def transform():
    log_message('Loading data')
    df = pd.read_csv('artifacts/ingested_data.csv')

    log_message('Checking duplicates')
    if df.duplicated().sum() != 0:
        log_message('Duplicates found')
        df.drop_duplicates(inplace=True)
        log_message('Dropping Duplicates')
    else:
        log_message('Dulicates are not found')
    
    log_message('Checking Missing Values')
    if df.isnull().sum().sum() != 0:
        log_message('Missing values are found')
        df.fillna(df.median(),inplace=True)
        log_message('Missing values are filled by median value')
    else:
        log_message('Missing values are not found')
    
    df.to_csv("artifacts/transformed_data.csv", index=False)
    log_message('Transformed data is saved')

    return df