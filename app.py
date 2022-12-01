#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Check Leave Availability", page_icon=None, layout='centered', initial_sidebar_state='auto')

# In[ ]:


@st.cache(allow_output_mutation=True)
def load(scaler_path, model_path):
    sc = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return sc , model


# In[ ]:


def inference(row, scaler, model, cols):
    df = pd.DataFrame([row], columns = cols)
    X = scaler.transform(df)
    features = pd.DataFrame(X, columns = cols)
    if (model.predict(features)==0):
        return "Yes"
    else: return "No"


# In[ ]:


st.title('Check Leave Availability')
st.write('An application that considers various parameters based on them, outputs whether a leave could be given to an employee.')
image = Image.open('data/diabetes_image.jpg')
st.image(image, use_column_width=True)
st.write('Please fill in the details of the employee under consideration in the left sidebar and click on the button below!')

age =           st.sidebar.number_input("CTC per annum", 1, 60, 25, 1)
pregnancies =   st.sidebar.number_input("Number Of Accidental Leaves taken", 0, 20, 0, 1)
glucose =       st.sidebar.slider("Casual Leave Spend", 0, 200, 25, 1)
skinthickness = st.sidebar.slider("Medical Leave Spend", 0, 99, 20, 1)
bloodpressure = st.sidebar.slider('Study Leave Spend', 0, 122, 69, 1)
insulin =       st.sidebar.slider("Earned Leave Spend", 0, 846, 79, 1)
bmi =           st.sidebar.slider("Weeks Spend in Leaves", 0.0, 67.1, 31.4, 0.1)
dpf =           st.sidebar.slider("Employee Improvement Index",  0.000, 2.420, 0.471, 0.001)

row = [pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]


# In[ ]:


if (st.button('Whether Leave can be assigned:')):
    cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    sc, model = load('models/scaler.joblib', 'models/model.joblib')
    result = inference(row, sc, model, cols)
    st.write(result)
