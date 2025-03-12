import os
import pickle
import pandas as pd
import mlflow
import dagshub
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

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

def load_model_and_scaler(model_path, scaler_path):
    """
    Load the saved model and scaler from pickle files.
    
    Args:
        model_path (str): Path to the saved model pickle file.
        scaler_path (str): Path to the saved scaler pickle file.
    
    Returns:
        model: Loaded model.
        scaler: Loaded scaler.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

def evaluate_and_log_predictions(model, scaler, X_test, y_test):
    """
    Evaluate the model, make predictions, and log results to MLflow.
    
    Args:
        model: Trained model.
        scaler: Scaler used for preprocessing.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
    """
    # Preprocess the test data
    X_test_scaled = scaler.transform(X_test)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log metrics to MLflow
    with mlflow.start_run():
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # Log predictions as a CSV file
        predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        predictions_df.to_csv("predictions.csv", index=False)
        mlflow.log_artifact("predictions.csv")

        # Log visualizations
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
        plt.savefig("actual_vs_predicted.png")
        mlflow.log_artifact("actual_vs_predicted.png")

        # Plot residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True)
        plt.xlabel("Residuals")
        plt.title("Residuals Distribution")
        plt.savefig("residuals_distribution.png")
        mlflow.log_artifact("residuals_distribution.png")

    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# Example usage
if __name__ == "__main__":
    # Load the saved model and scaler
    model, scaler = load_model_and_scaler("models/trainedModel.pkl", "models/scaler.pickle")

    # Load test data (replace with your actual test data)
    test_data = pd.read_csv("data/test.csv")  # Replace with your test data path
    X_test = test_data.drop('quality', axis=1)
    y_test = test_data['quality']

    # Evaluate and log predictions
    evaluate_and_log_predictions(model, scaler, X_test, y_test)