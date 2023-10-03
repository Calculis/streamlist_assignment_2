
import streamlit as st
import numpy as np 
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle


# Load model
with open('knn_weather.pkl', 'rb') as file:
    model, weather_encoder = pickle.load(file)

# Streamlit app
st.title("Weather Prediction")

# Get user input for each variable
precipitation_input = st.slider('Enter precipitation :',0.0,60.0)
temp_max_input = st.number_input('Enter max temperature in F (-20 to 50):', min_value=-20.0, max_value=50.0)
temp_min_input = st.number_input('Enter min temperature in F (-20 to 50):', min_value=-20.0, max_value=50.0)
wind_input = st.slider('Enter wind speed:', 0.0, 20.0)


# Create a DataFrame with user input
x_new = pd.DataFrame({
    'precipitation': [precipitation_input],
    'temp_max': [temp_max_input],
    'temp_min': [temp_min_input],
    'wind': [wind_input]
})


# Prediction
def predict():
    y_pred_new = model.predict(x_new)
    result = weather_encoder.inverse_transform(y_pred_new)
    return result

# Display result
st.subheader('Prediction Result:')
button_clicked = st.button("Enter")
if button_clicked:
    prediction = predict()
    st.write(f"Predict Weather: {prediction[0]}")
