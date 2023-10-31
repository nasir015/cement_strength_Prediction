import mysql.connector
import pandas as pd
import os
from pipeline.logging import logger
path = open("E:\\Neoron\\Programming_Practice\\Machine_Learning_Project\\cement_strength_reg\\Log\\retrieve_from_sql.txt", "w")
log_path= 'E:\\Neoron\\Programming_Practice\\Machine_Learning_Project\\cement_strength_reg\\Log\\retrieve_from_sql.txt'


def retrieve_data_from_mysql_table(database, table):
    logger(log_path,'Connect to MySQL server')
    cnx = mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password='4317',
        auth_plugin='mysql_native_password',
        database=database
    )

    logger(log_path,'Create cursor')
    cursor = cnx.cursor()

    logger(log_path,'Execute SELECT query')
    query = f"SELECT * FROM {table}"
    cursor.execute(query)

    logger(log_path,'Fetch all rows')
    rows = cursor.fetchall()

    logger(log_path,'Get column names')
    column_names = cursor.column_names


    logger(log_path,'create a dataframe with the column names as the header')

    df = pd.DataFrame(rows, columns=column_names)

    logger(log_path,'create an directory and then save the dataframe as a csv file')
    path = os.getcwd()
    path = path + '/dataset'
    os.mkdir(path)

    df.to_csv(path + '/data.csv', index=False)

    logger(log_path,'Close cursor and connection')
    cursor.close()
    cnx.close()

    return df



if __name__ == "__main__":
    data = retrieve_data_from_mysql_table('cement_strength_prediction' ,'dataset')
    print(data)
