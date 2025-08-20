import os 
import sys 
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import CustomException


def save_object(file_path: str, obj) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        tmp_path = file_path + ".tmp"
        with open(tmp_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
        os.replace(tmp_path, file_path)
    except Exception as e:
        raise CustomException(f"Failed to save object to {file_path}: {e}", sys) from e
    

def evaluate_model(x_train, y_train, x_test, y_test, models: dict) -> dict:
    """
    Trains and evaluates multiple models using R2 score.
    Returns: {model_name: score}
    """
    try:
        report = {}

        for name, model in models.items():
            model.fit(x_train, y_train)

            y_test_pred = model.predict(x_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[name] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(f"Error while evaluating models: {e}", sys) from e
