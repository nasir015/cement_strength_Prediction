import sys
import os
import csv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline.logging import logger
from src.pipeline.Exception import CustomException
import pandas as pd
import mysql.connector as connection
path = open("E:\\Neoron\\Programming_Practice\\Machine_Learning_Project\\cement_strength_reg\\Log\\table_creation&upload_data_sql.txt", "w")
log_path= 'E:\\Neoron\\Programming_Practice\\Machine_Learning_Project\\cement_strength_reg\\Log\\table_creation&upload_data_sql.txt'


# create an function to create database in mysql

def create_database_sql(database_name):
    try:
        logger(log_path,'Create a connection to your MySQL server')
        conn = connection.connect(port='3306',
                                host='127.0.0.1',
                                user ='root',
                                password ='4317',
                                auth_plugin='mysql_native_password')
        cursor = conn.cursor()

        logger(log_path,'Create a new database')
        cursor.execute(f"CREATE DATABASE {database_name}")

        logger(log_path,'Close the connection')
        conn.close()
        logger(log_path,'Database created')

    except Exception as e:
            logger(log_path,e)
            raise CustomException(e,sys)


# create function to upload data from csv to sql

def table_creation_sql(data,database_name,table_name):
    try:
        
            
        logger(log_path,'Read the CSV file to identify the columns and data types')
        csv_file = data   # Replace with the path to your CSV file
        df = pd.read_csv(csv_file)  # Read a sample of rows to infer data types

        logger(log_path,'Extract column names and data types')
        columns = df.columns.tolist()
        data_types = [df[col].dtype for col in df.columns]

        logger(log_path,'Create a SQL CREATE TABLE statement')
        table_name = table_name  # Replace with your desired table name
        create_table_sql = f"CREATE TABLE {table_name} ("

        for col, dtype in zip(columns, data_types):
            if dtype == 'int64':
                column_definition = f"{col} INT"
            elif dtype == 'float64':
                column_definition = f"{col} FLOAT"
            else:
                max_length = df[col].str.len().max()
                column_definition = f"{col} VARCHAR({max_length})"
            
            create_table_sql += f"{column_definition}, "

        logger(log_path,'Remove the trailing comma and close the statement')
        create_table_sql = create_table_sql.rstrip(', ')
        create_table_sql += ");"

        logger(log_path,'Create a connection to your MySQL database')
        conn = connection.connect(port='3306',
                                host='127.0.0.1',
                                user ='root',
                                password ='4317',
                                auth_plugin='mysql_native_password',
                                database= database_name)

        cursor = conn.cursor()

        cursor.execute(create_table_sql)
        
        logger(log_path, 'Table created')


    except Exception as e:
            logger(log_path,e)
            raise CustomException(e,sys)

def data_upload_sql(csv_file,database_name, table_name):

    try:
        logger(log_path,'Create a connection to your MySQL database')
            
        
        conn = connection.connect(port='3306',
                        host='127.0.0.1',
                        user ='root',
                        password ='4317',
                        auth_plugin='mysql_native_password',
                        database=database_name)
        cursor = conn.cursor()

        logger(log_path,'Insert data into the table')

        with open(csv_file, 'r') as file:
                
                csv_reader = csv.reader(file)
                header = next(csv_reader)  # Read the header
                for row in csv_reader:
                    insert_sql = f"INSERT INTO {table_name} ({', '.join(header)}) VALUES ({', '.join(['%s'] * len(header))})"
                    cursor.execute(insert_sql, row)
                    conn.commit()
        logger(log_path,'Data imported')

    except Exception as e:
            logger(log_path,e)
            raise CustomException(e,sys)


    
    

if __name__ == '__main__':
    #table_creation_sql('E:\\Neoron\\Programming_Practice\\Machine_Learning_Project\\cement_strength_reg\\Data\\Concrete_Compressive_Strength.csv','cement_strength_prediction','dataset')
    data_upload_sql('E:\\Neoron\\Programming_Practice\\Machine_Learning_Project\\cement_strength_reg\\Data\\Concrete_Compressive_Strength.csv','cement_strength_prediction','dataset')
    #create_database_sql('cement_strength_prediction')


    #E:\Neoron\Programming_Practice\Machine_Learning_Project\cement_strength_reg\Data\Concrete Compressive Strength.csv