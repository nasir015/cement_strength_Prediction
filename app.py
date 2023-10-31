import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.logging import logger
from src.pipeline.Exception import CustomException
from src.utils.common import load_object


st.title('''Cement Strength Regression''')


def user_input_features():
    Cement = st.number_input('Cement(102.0 - 540.0)'),
    Blast_Furnace_Slag = st.number_input('Blast Furnace Slag(0.0 - 359.4)'),
    Fly_Ash = st.number_input('Fly Ash0.0 - 200.1)'),
    Water = st.number_input('Water(121.8 - 247.0)'),
    Superplasticizer = st.number_input('Superplasticizer(0 - 32.2)'),
    Coarse_Aggregate = st.number_input('Coarse Aggregate(801.0 - 1145.0)'),
    Fine_Aggregate = st.number_input('Fine Aggregate(594.0 - 992.6)'),
    Age_day = st.number_input('Age(1 - 365)')

    data = {'Cement': Cement,
            'Blast_Furnace_Slag_': Blast_Furnace_Slag,
            'Fly_Ash': Fly_Ash,
            'Water': Water,
            'Superplasticizer': Superplasticizer,
            'Coarse_Aggregate': Coarse_Aggregate,
            'Fine_Aggregate': Fine_Aggregate,
            'Age_day': Age_day}

    features = pd.DataFrame(data, index=[0])

    return features


input_df = user_input_features()

if st.button('Prediction'):
    def predict(feature):
            
        try:
            model_path = 'artifacts\\model.pkl'
            preprocessor_path = 'artifacts\\preprocessor.pkl'
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_scaled = preprocessor.transform(feature)
            prediction = model.predict(data_scaled)
            
            return prediction
            
        except Exception as e:
            raise CustomException(e,sys)
        
    prediction = predict(input_df)

    st.write( 'Concrete compressive strength is: ',prediction[0])


df = pd.read_csv('dataset\\data.csv')

st.subheader('Data Information:')

# create a heatmap to see the correlation between the features

if st.checkbox('Show raw data'):
    st.write(df)

st.write(df.describe())


# create heatmap
st.subheader('Correlation Matrix:')

st.set_option('deprecation.showPyplotGlobalUse', False)
plt.figure(figsize=(12,12))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
st.pyplot()


