import os, sys
import pickle
import pandas as pd
import mlflow
import dagshub
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Initialize DAGsHub
dagshub.init(repo_owner='chandu1613',
             repo_name='MLProject',
             mlflow=True)

# Set MLflow tracking credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = "Chandu1613"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "6656b403fc3d6206838ca3affb4b3b2b84d84b8a"

# Set MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/Chandu1613/MLProject.mlflow")

# Set experiment name
mlflow.set_experiment("Wine_Prediction_Evaluation")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logs import log_message

log_message("Loading trained model....")

def load_model(model_path):
    """
    Load the saved model and scaler from pickle files.
    
    Args:
        model_path (str): Path to the saved model pickle file.
    
    Returns:
        model: Loaded model.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def evaluate_and_tuning(model,train,test):
    """
    Evaluate the model, make predictions, and log results to MLflow.
    
    Args:
        model: Trained model.
        scaler: Scaler used for preprocessing.
        test (pd.DataFrame): Test features.
    """

    X_train = train.drop('quality', axis=1)
    X_test = test.drop('quality', axis=1)
    y_train = train['quality']
    y_test = test['quality']

    model_name = type(model).__name__

    param_grids = {
        "RandomForestRegressor": {
            "n_estimators": [100, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False]
        },
        "GradientBoostingRegressor": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 10],
            "subsample": [0.7, 0.8, 0.9, 1.0]
        },
        "SVR": {
            "kernel": ["linear", "rbf", "poly"],
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto"]
        },
        "ElasticNet": {
            "alpha": [0.01, 0.1, 0.5, 1.0, 10],
            "l1_ratio": [0.2, 0.5, 0.7, 0.9]
        },
    }

    log_message("Hyper perameter tuning based on the model....")
    if model_name == "LinearRegression":
        print("Linear Regression has no hyperparameters to tune.")
        best_model_tuned = Ridge(alpha=1.0)
        best_model_tuned.fit(X_train, y_train)

    elif model_name in param_grids:
        param_grid = param_grids[model_name]
        tuner = RandomizedSearchCV(
            estimator=model, param_distributions=param_grid,
            n_iter=10, cv=5, scoring="r2", n_jobs=-1, verbose=2, random_state=42
        )
        
        tuner.fit(X_train, y_train)
        best_model_tuned = tuner.best_estimator_
        print(f"Best Tuned Parameters: {best_model_tuned.get_params()}")

    else:
        print("No tuning strategy defined for this model.")
        best_model_tuned = model
    
    log_message("prediction before the tuning.......")
    y_pred_before = model.predict(X_test)
    log_message("prediction after the tuning.......")
    y_pred_after = best_model_tuned.predict(X_test)

    log_message("Metrics after the tuning.......")
    rmse_before = mean_squared_error(y_test, y_pred_before)
    r2_before = r2_score(y_test, y_pred_before)
    
    log_message("Metrics before the tuning.......")
    rmse_after = mean_squared_error(y_test, y_pred_after)
    r2_after = r2_score(y_test, y_pred_after)

    log_message("Logging metrics.... ")

    with mlflow.start_run():
        mlflow.log_metric("RMSE", rmse_before)
        mlflow.log_metric("R2", r2_before)

        log_message("Saving the untuned prediction.csv")
        predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_before})
        predictions_df.to_csv("artifacts/predictions.csv", index=False)
        mlflow.log_artifact("artifacts/predictions.csv")

        log_message("Saving the untuned Actual vs Predicted plot.....")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred_before)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.savefig("images/actual_vs_predicted.png")
        mlflow.log_artifact("images/actual_vs_predicted.png")

        log_message("Saving the untuned Residuals Distribution plot.....")
        residuals = y_test - y_pred_before
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True)
        plt.xlabel("Residuals")
        plt.title("Residuals Distribution")
        plt.savefig("images/residuals_distribution.png")
        mlflow.log_artifact("images/residuals_distribution.png")

        mlflow.log_metric("RMSE", rmse_after)
        mlflow.log_metric("R2", r2_after)

        log_message("Saving the tuned prediction.csv")
        predictions_df_after = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_after})
        predictions_df_after.to_csv("artifacts/tuned_prediction.csv", index=False)
        mlflow.log_artifact("artifacts/tuned_prediction.csv")

        log_message("Saving the tuned Actual vs Predicted plot.....")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred_before)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Tuned Actual vs Predicted")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.savefig("images/tuned_actual_vs_predicted.png")
        mlflow.log_artifact("images/tuned_actual_vs_predicted.png")

        log_message("Saving the tuned Residuals Distribution plot.....")
        residuals = y_test - y_pred_before
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True)
        plt.xlabel("Residuals")
        plt.title("Tuned Residuals Distribution")
        plt.savefig("images/residuals_distribution.png")
        mlflow.log_artifact("images/tuned_residuals_distribution.png")

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, "Best_Model")

    log_message("Saving the Tuned Model......")
    os.makedirs("models", exist_ok=True)
    with open("models/tunedModel.pkl", "wb") as f:
        pickle.dump(best_model_tuned, f)

    