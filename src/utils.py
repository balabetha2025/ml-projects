import os 
import sys 
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV


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
    

def evaluate_model(x_train, y_train, x_test, y_test, models,param):
   
    try:
        report = {}

        for i in range(len(list(models))):
            model= list(models.values())[i] 
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3,n_jobs=-1,verbose=1,refit=True)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            y_test_pred = model.predict(x_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = (test_model_score,gs.best_params_)

        return report
    
    except Exception as e:
        raise CustomException(f"Error while evaluating models: {e}", sys) from e
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(f"Failed to load object from {file_path}: {e}", sys) from e
        