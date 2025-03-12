import os
import sys
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


dagshub.init(repo_owner='chandu1613',
             repo_name='MLProject',
             mlflow=True)

os.environ["MLFLOW_TRACKING_USERNAME"] = "Chandu1613"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "6656b403fc3d6206838ca3affb4b3b2b84d84b8a"

mlflow.set_tracking_uri("https://dagshub.com/Chandu1613/MLProject.mlflow")

mlflow.set_experiment("Wine_Prediction")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logs import log_message

def training(train, test):
    log_message("Loading the train and test data.....")

    train_data = train
    test_data = test

    log_message("Splitting the data.....")

    X_train = train_data.drop('quality', axis=1)
    X_test = test_data.drop('quality', axis=1)
    y_train = train_data['quality']
    y_test = test_data['quality']

    log_message("Model Training....")

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "LinearRegression": LinearRegression(),
        "SVC": SVR(),
        "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.7, random_state=101)
    }

    best_model = None
    best_score = float('-inf')

    log_message("Training models...")

    for name, model in models.items():
        try:
            with mlflow.start_run():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                log_message(f"{name} -> MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                mlflow.log_param("model", name)
                mlflow.log_metric("MSE", mse)
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("R2", r2)
                mlflow.sklearn.log_model(model, "model")

                if r2 > best_score:
                    best_model = model
                    best_score = r2
        except Exception as e:
            log_message(f"Error training {name}: {str(e)}")

    log_message("Saving the best model......")

    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)
    with open("models/trainedModel.pkl", "wb") as f:
        pickle.dump(best_model, f)

    log_message("Model training is done..")