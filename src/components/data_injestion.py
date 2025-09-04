import pandas as pd
import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformConfig
from src.components.data_transformation import DataTransformation

@dataclass
class DataInjestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    
class DataInjestion:
    def __init__(self):
        self.injestion_config = DataInjestionConfig()
        
    def initiate_data_injestion(self):
        logging.info("Entered the data injestion method")
        try:
            df = pd.read_csv("notebooks/stud.csv")
            logging.info("Read the dataset as data frame")
            os.makedirs(os.path.dirname(self.injestion_config.train_data_path), exist_ok = True)
            
            logging.info("train_test_split initiated")
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state=42)
            train_set.to_csv(self.injestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.injestion_config.test_data_path, index = False, header = True)
            logging.info("injestion of data is completed")
            return(
                self.injestion_config.train_data_path,
                self.injestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataInjestion()
    train_data, test_data = obj.initiate_data_injestion()
    
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)