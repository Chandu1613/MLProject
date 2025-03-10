import os
import sys
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logs import log_message

def preprocessing(df):
    df_capped = df.copy() 
    numeric_cols = df.select_dtypes(include=['number']).columns

    log_message("Outliers are Handling using IQR method.....")

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_capped[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    log_message("Completed the Outlier Handling.....")

    X = df_capped.drop('quality',axis=1)
    y = df_capped['quality']

    X_numeric_cols = X.select_dtypes(include=['number']).columns

    scaler = StandardScaler()

    X_scaled = X.copy()
    X_scaled[X_numeric_cols] = scaler.fit_transform(X_scaled[X_numeric_cols])

    log_message("Scaling is done using StandardScaler method")

    df_scaled =pd.concat([X_scaled,y],axis=1)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    log_message("Saving the Scaler Model.")
    
    train,test = train_test_split(df_scaled,test_size=0.25,random_state=101)
    log_message("Splitting the train and test data in the ratio of 3:1")

    os.makedirs("artifacts", exist_ok=True)
    train.to_csv("artifacts/train.csv", index=False)
    test.to_csv("artifacts/test.csv", index=False)
    log_message('Train and Test data is saved')
    
    return train,test
