# Streamlit
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import json
import requests

st.write("""
# Hello world
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    PU = st.sidebar.text_input("PU Location ID")
    DO = st.sidebar.text_input("DO Location ID")
    trip_distance = st.sidebar.number_input("Trip Distance")


    data = {'PULocationID': PU,
            'DOLocationID': DO,
            'trip_distance': trip_distance}
    
    features = data
    return features

df_dict = user_input_features()

if st.button("Predict"):
    response = requests.post(
        url="http://127.0.0.1:4444/predict",
        data=json.dumps(df_dict) 
    )

    st.write(response.text) 


