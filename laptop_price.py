import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("laptop price prediction")

data = pd.read_csv("clean_laptop_data.csv")
data
df_train, df_test = train_test_split(data, train_size = 0.85, test_size = 0.15, random_state = 1)
X_train = df_train[['Ram','Weight','Gaming','Ultrabook','Workstation','Intel','Nvidia','Mac','Windows']]
y_train = df_train['Price'].astype('int')
lm = LinearRegression()
lr_model = lm.fit(X_train, y_train)

# ############ Model Creation ##################
with st.form(key='my_form'):
        Ram = st.number_input('Ram')
        Weight = st.number_input('Weight')
        # HDD = st.number_input('HDD')
        Gaming = st.number_input('Gaming')
        # Netbook = st.number_input('Netbook')
        # Notebook = st.number_input('Notebook')
        Ultrabook = st.number_input('Ultrabook')
        Workstation = st.number_input('Workstation')
        Intel = st.number_input('Intel')
        Nvidia = st.number_input('Nvidia')
        Mac= st.number_input('Mac')
        Windows = st.number_input('Windows')

        df = pd.DataFrame({'Ram':[Ram],'Weight':[Weight],'Gaming':[Gaming],'Ultrabook':[Ultrabook],'Workstation':[Workstation],'Intel':[Intel],'Nvidia':[Nvidia],'Mac':[Mac],'Windows':[Windows]})
    
        if st.form_submit_button():
           output = lr_model.predict(df)
           output= output[0]
           st.write("predicted price is:",output)

