import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from pipeline.logging import logger
from pipeline.Exception import CustomException
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

path = open("E:\\Neoron\\Programming_Practice\\Machine_Learning_Project\\cement_strength_reg\\Log\\common.txt", "w")
log_path= 'E:\\Neoron\\Programming_Practice\\Machine_Learning_Project\\cement_strength_reg\\Log\\common.txt'


logger(log_path,'save_object function is started')
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
    logger(log_path,'save_object function is completed')




logger(log_path,'evaluate_models function is started')

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            logger(log_path,'evaluate_models function is completed')

        return report
    

    except Exception as e:
        raise CustomException(e, sys)
    
logger(log_path,'load_object function is started')   


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
        

    except Exception as e:
        raise CustomException(e, sys)
logger(log_path,'load_object function is completed')

