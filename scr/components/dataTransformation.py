import os
import sys
from scr.exception import CustomException
from scr.logger import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scr.utiles import save_obj
from dataclasses import dataclass

@dataclass
class dataTransformerConfig:
    preprosessor_obj_path = os.path.join("artifacts", "dataTransformer.pkl")

class dataTransformer:

    def __init__(self):
        self.data_transformer_config_obj = data_transformer_config()
    def get_data_transformer_obj(self):
        try:
            numerical_features = ["writing score", "reading_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            numerical_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("impuer", SimpleImputer(strategy="median"))
            ]
        )
            categorical_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=False)),
                    ("impuer", SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder", OneHotEncoder())
                    ]
        )
            
            preprosessor = ColumnTransformer(
                [
                    ("numerical_transformer", numerical_pipeline, numerical_features),
                    ("categorical_transformer", categorical_pipeline, categorical_features)
            ]
        )

            return preprosessor
        
        except Exception as e:
            CustomException(e, sys)
    
    def data_transformer_intiation(self, train_path, test_path):
        try:
            train_df = pd.read_csv("train_path")
            test_df = pd.read_csv("test_path")
            preprosessor_obj = self.get_data_transformer_obj()
            target_column = "math_score"
            input_features_train_df = train_df.drop(columns=[target_column], axis=1)
            input_features_test_df = test_df.drop(columns=[target_column], axis=1)
            output_features_train_df = train_df["math_score"]
            output_features_test_df = test_df["math_score"]
            input_features_train_arr = preprosessor_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprosessor_obj.transform(input_features_test_df)
            train_arr = np.c_[
                input_features_train_arr, np.array(output_features_train_df)
            ]

            test_arr = np.c_[
                input_features_test_arr, np.array(output_features_test_df)
            ]

            save_obj(
                file_path=self.data_transformer_config_obj.preprosessor_obj_path,
                obj=preprosessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformer_config_obj.preprosessor_obj_path
            )
        except Exception as e:
            CustomException(e, sys)
        
    
