import os
import sys
from scr.exception import CustomException
from scr.logger import logging
import pandas as pd
import numpy as np
import pickle
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_obj(self, file_path, obj):
    try:
        dirt_path = os.path.dirname(file_path)
        os.makedirs("dirt_path", exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        CustomException(e, sys)

def evaluate_model(self, x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            param = [list(params.keys())][i]
            gv = GridSearchCV(model, param, cv=3)
            gv.fit(x_train, y_train)
            model.set_params(**gv.best_params_)
            model = model.fit(x_train, y_train)
            y_train_predict = model.predict(x_train)
            y_test_predict = model.predict(x_test)
            r2_score_train = r2_score(y_train, y_train_predict)
            r2_score_test = r2_score(y_train, y_test_predict)
            report[list(models.keys())[i]] = r2_score_test
            return report

    except Exception as e:
        CustomException(e, sys)

def load_file(self, file_path):
    try:
        with open(self.file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        CustomException(e, sys)
    
