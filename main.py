import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.data_ingestion import load_data
from src.data_transformation import transform
from src.data_preprocessing import preprocessing
from src.model_trainer import training
from src.model_evaluation import load_model, evaluate_and_tuning

if __name__ == "__main__":
    dataset_path = "datasets/winequality.zip"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found!")

    df = load_data(dataset_path)
    
    transformed = transform()

    train,test = preprocessing(transformed)

    training(train,test)

    model = load_model('models/trainedModel.pkl')

    evaluate_and_tuning(model,train,test)