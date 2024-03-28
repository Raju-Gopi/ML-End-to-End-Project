import sys
import pandas as pd
from scr.exception import CustomException
import os
from scr.utiles import load_file


class pridictionPipeline:
    def __init__(self) -> None:
        pass
    def predict(self, features):
        try:
            preprossor_model_path = os.path.join("artifacts", "preprossor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")
            preprossor = load_file(preprossor_model_path)
            model = load_file(model_path)
            preprossor_df = preprossor.transform(self.features)
            predicts = model.predict(preprossor_df)
            return predicts
        except Exception as e:
            CustomException(e, sys)

class customData:

    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_df(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            CustomException(e, sys)
