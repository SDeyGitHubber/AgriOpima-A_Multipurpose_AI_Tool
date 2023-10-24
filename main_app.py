import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model
clf_model = joblib.load('models/classification_model.pkl')
reg_model = joblib.load('models/yield_predictor.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')
scaler = joblib.load('models/scaler.pkl')
scaler_clf = joblib.load('models/scaler_clf.pkl')

# Taking input dataframe
df1 = pd.read_csv('Datasets/Crop_Yield_dataset/prty.csv')

# Define the Streamlit app
st.title("Crop Recommendation and Yield Prediction App")

# Create a top navigation menu
mode = st.selectbox("", ["Home", "Crop Recommendation", "Crop Yield Prediction"])

# @st.cache_data(allow_output_mutation=True, hash_funcs={pd.DataFrame: id})
def predict_yield(user_input):
    # Perform preprocessing, including label encoding and scaling
    categorical_cols = ['Item', 'Area', 'Element']
    for col in categorical_cols:
        le = label_encoders[col]
        user_input[col] = le.transform(user_input[col])  # Transform the column in-place
    
    user_input = scaler.transform(user_input)  # Ensure the feature order is consistent

    # Predict the yield
    prediction = reg_model.predict(user_input)
    return prediction

# @st.cache_data(allow_output_mutation=True, hash_funcs={pd.DataFrame: id})
def recommend_crop(user_input):
    # Preprocess user input data and run yield prediction logic
    user_input = scaler_clf.transform(user_input)

    # Use the loaded model for yield prediction
    prediction = clf_model.predict(user_input)
    
    return prediction


if mode == "Home":
    st.write("Welcome to the Crop Recommendation and Yield Prediction App!")

elif mode == "Crop Yield Prediction":
    # Add user input components for crop recommendation
    crop = st.selectbox("Crop:", df1['Item'].unique())
    country = st.selectbox("Country:", df1['Area'].unique())
    element = st.selectbox("Element:", df1['Element'].unique())
    avg_temp = st.number_input("Average Temperature (°C):")
    avg_rainfall_mm_per_year = st.number_input("Average Rainfall (mm per year):")
    pesticides_tonnes = st.number_input("Pesticides (tonnes):")

# Create a user input DataFrame
    user_input = pd.DataFrame({
        'Area': [country],
        'Element': [element],
        'Item': [crop],
        'pesticides_tonnes': [pesticides_tonnes],
        'average_rain_fall_mm_per_year': [avg_rainfall_mm_per_year],
        'avg_temp': [avg_temp],
    })

     # Ensure the feature order is consistent
    if st.button("Predict Yield"):
        prediction = predict_yield(user_input)
        st.write('The yield prediction is', prediction)

elif mode == "Crop Recommendation":
    # Add user input components for yield prediction
    N = st.number_input("Nitrogen requirements:")
    P = st.number_input("Phosphorus requirements:")  # Add all available crop options
    K = st.number_input("Potassium requirements:")  # Add all available country options
    temerature = st.number_input("Average Temperature (°C):")
    humidity = st.number_input("Humidity: ")
    ph = st.number_input("Ph value of soil: ")
    rainfall = st.number_input('Annual Rainfall (mm per year): ')

# Preprocess the user input data
    user_input = pd.DataFrame({
    'N': [N],
    'P': [P],
    'K': [K],
    'temperature': [temerature],
    'humidity': [humidity],
    'ph': [ph],
    'rainfall':[rainfall]
    })

    if st.button("Recommend Crop"):
        prediction = recommend_crop(user_input)
        st.write('The recommended crop is', prediction)
