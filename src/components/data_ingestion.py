import os
import sys
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle 

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self) -> Tuple[str, str]:
        
        logging.info("Started data ingestion process")
        try:
            df = pd.read_csv('notebook/DATA/Sleep_health_and_lifestyle_dataset.csv')
            logging.info('Dataset read into DataFrame')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f'Raw data saved to {self.ingestion_config.raw_data_path}')
            
            logging.info("Performing train-test split")
            train_set, test_set = train_test_split(df, train_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data ingestion complete")
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
            
            
        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)
            
if __name__ == "__main__":
    # Initialize DataIngestion
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    
    # Initialize DataTransformation
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    
    # # Optional: Save the preprocessing object for later use
    # preprocessing_obj_path = os.path.join('artifacts', 'preprocessing_obj.pkl')
    # with open(preprocessing_obj_path, 'wb') as file:
    #     pickle.dump(preprocessing_obj, file)
    # logging.info(f'Preprocessing object saved to {preprocessing_obj_path}')
    
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))