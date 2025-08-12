import os 
import sys 
import dill

from src.exception import CustomException

def save_object(file_path:str,obj)->None:
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
        tmp_path = file_path + ".tmp"
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
        os.replace(tmp_path, file_path)
    
    except Exception as e:
            raise CustomException(f"Failed to save object to {file_path}: {e}", sys) from e