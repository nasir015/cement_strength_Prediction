import os 
import sys
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pipeline.logging import logger
from pipeline.Exception import CustomException



path = open("E:\\Neoron\\Programming_Practice\\Machine_Learning_Project\\cement_strength_reg\\Log\\data_transformation.txt", "w")
log_path= 'E:\\Neoron\\Programming_Practice\\Machine_Learning_Project\\cement_strength_reg\\Log\\data_transformation.txt'

logger(log_path,'Importing all libraries')

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    
    logger(log_path,'DataTransformationConfig class started')
class DataTransformation:
    
    def __init__(self):
        self.Data_transformation_config = DataTransformationConfig()
        
    logger(log_path,'DataTransformation class started')

    def get_data_transformar_obj(self):
        try:
            numerical_features = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
            
            logger(log_path, 'Numerical and categorical features are defined')
            
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', StandardScaler())
            ])
            logger(log_path, 'num_pipeline is defined')
        
            
            
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_features)
            ])
            logger(log_path, 'Column transformation is defined')
            
            logger(log_path, 'DataTransformation class ended')
            return preprocessor
        
        
        
        except Exception as e:
            logger(log_path, 'Exception occured in get_data_transformar_obj method of DataTransformation class')
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,train_data_path,test_data_path):
        logger(log_path, 'initiate_data_transformation method of DataTransformation class started')
        try:
            
            train_df = pd.read_csv(train_data_path)
            logger(log_path, 'Train data is read')
            test_df = pd.read_csv(test_data_path)
            logger(log_path, 'Test data is read')
            
            
            preprocessing_obj = self.get_data_transformar_obj()
            logger(log_path, 'preprocessing_obj is called')
            
            target_column_name = "quality"
            logger(log_path, 'Target column is defined')
            
            input_feature_train = train_df.drop(target_column_name,axis=1)
            terget_feature_train = train_df[target_column_name]
            logger(log_path, 'Input and target features are defined for train data')
            
            input_feature_test = test_df.drop(target_column_name,axis=1)
            terget_feature_test = test_df[target_column_name]
            logger(log_path, 'Input and target features are defined for test data')
            
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train)
            logger(log_path, 'Input feature train data is transformed')
            
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test)
            logger(log_path, 'Input feature test data is transformed')
            
            
            
            train_arr = np.c_[input_feature_train_arr,
                              np.array(terget_feature_train)]
            logger(log_path, 'Train data is concatenated')
            
            
            test_arr = np.c_[input_feature_test_arr,
                             np.array(terget_feature_test)]
            logger(log_path, 'Test data is concatenated')
            
            
            save_object (
                file_path = self.Data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logger(log_path , 'save_object function is called')
            
            return (
                train_arr,
                test_arr,
                self.Data_transformation_config.preprocessor_obj_file_path
            )
            
            
        except Exception as e:
            logger(log_path, 'Exception occured while executing initiate_data_transformation method of DataTransformation class')
            raise CustomException(e,sys)
        


if __name__ == "__main__":
    a = DataTransformation()
    a.initiate_data_transformation('E:\\Neoron\\Programming_Practice\\Machine_Learning_Project\\cement_strength_reg\\artifacts\\train.csv','E:\\Neoron\\Programming_Practice\\Machine_Learning_Project\\cement_strength_reg\\artifacts\\test.csv')
