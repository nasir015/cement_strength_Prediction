import os
import sys
sys.path.append( os. path. dirname( os. path. dirname( os. path. abspath(__file__))))
from dataclasses import dataclass
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from pipeline.Exception import CustomException
from pipeline.logging import logger
from utils.common import save_object,evaluate_models
from components.data_preprocessing import DataTransformation,DataTransformationConfig
path = open("E:\\Neoron\\Programming_Practice\\Machine_Learning_Project\\cement_strength_reg\\Log\\model_training.txt", "w")
log_path= 'E:\\Neoron\\Programming_Practice\\Machine_Learning_Project\\cement_strength_reg\\Log\\model_training.txt'


logger(log_path,'Importing all libraries')


@dataclass
class ModelTrainerConfig:
    train_model_file_path  = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
        
    def initiate_model_trainer(self,train_array, test_array):
        
        try:
            logger(log_path, 'spliting train and test data')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logger(log_path, 'train and test data spliting is done')
            '''
            train_array[:,:-1] extracts all the columns from train_array except for the last one,
            which is assumed to be the labels. This is your training data.
            
            train_array[:,-1] extracts only the last column from train_array, 
            which is assumed to be the labels for the training data.
            
            test_array[:,:-1] extracts all the columns from test_array except 
            for the last one, which is assumed to be the labels.
            This is your testing data.
            
            test_array[:,-1] extracts only the last column from test_array, 
            which is assumed to be the labels for the testing data.
            
            '''
            logger(log_path, 'Model specification is started')
            model = {
                'XGBRegressor': XGBRegressor()
            }
            logger(log_path, 'Model specification is done')
            
            params={
                "XGBRegressor":{
                    'learning_rate':[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,0.4,0.5],
                    'n_estimators': [1,2,4,8,16,32,64,128,256],
                    'max_depth': [1, 2, 4, 8, 16],
                }
            }
            
            model_report:dict = evaluate_models(X_train= X_train,
                                               y_train= y_train,
                                               X_test= X_test,
                                               y_test= y_test,
                                               models= model,
                                               param = params
                                               )
            
            
            # To get the best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
            logger(log_path, 'To get the best model score from dictionary')
            
            # To get the best model name from dictionary
            logger(log_path, 'To get the best model name from dictionary')
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = model[best_model_name]
            logger(log_path, 'best model is selected')
            
            logger(log_path, 'best model is trained')
            
            
            
            
            
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
                )
            logger(log_path, 'Model is saved')    
                
            
            
            predicted = best_model.predict(X_test)
            logger(log_path, 'Model is predicted')
            
            r2 = r2_score(y_test,predicted)
            logger(log_path, 'R2 score is calculated')
            
            return r2
        
            
            
             
        except Exception as e:
            logger(log_path, 'Exception occured in initiate_model_trainer method of ModelTrainer class')
            raise CustomException(e,sys)
        
        

