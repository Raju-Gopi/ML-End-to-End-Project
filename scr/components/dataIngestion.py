import os
import sys
from scr.logger import logging
from scr.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from scr.components.dataTransformation import dataTransformer
from scr.components.modelTrainer import ModelTrainer

@dataclass
class data_ingestion_config:
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join("artifacts", "raw.csv")

class data_ingestion:
    logging("Data ingestion started")
    def __init__(self):
        self.ingestion_config = data_ingestion_config()
    def data_ingestion_intiate(self):
        logging("Data ingestion intiated")
        try:
            df = pd.read_csv("scr\notebook\data\stud.csv")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            logging("Data ingestion completed")
        except Exception as e:
            CustomException(e, sys)

if __name__ == "__main__":
    obj = data_ingestion()
    train_set, test_set = obj.data_ingestion_intiate()
    data_transformation = dataTransformer()
    train_arr, test_arr, _= data_transformation.data_transformer_intiation(train_set, test_set)
    model_trained = ModelTrainer()
    model_trained.intiate_model_training(train_arr=train_arr, test_arr=test_arr)
    

            

        
