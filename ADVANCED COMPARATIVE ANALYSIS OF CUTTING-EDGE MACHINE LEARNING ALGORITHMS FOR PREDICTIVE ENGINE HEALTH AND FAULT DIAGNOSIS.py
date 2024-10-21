import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

# Load models (use st.cache_resource to cache models as they're large and resource-heavy)
@st.cache_resource
def load_model_files():
    with open("knn_model.pkl", 'rb') as file:
        knn = pickle.load(file)
    with open("svm_model.pkl", 'rb') as file:
        svm = pickle.load(file)
    with open("random_forest_model.pkl", 'rb') as file:
        rdf = pickle.load(file)
    with open("xgboost_model.pkl", 'rb') as file:
        xgb = pickle.load(file)
    with open("gradient_boosting_model.pkl", 'rb') as file:
        gbc = pickle.load(file)
    lstm = load_model("lstm_model.h5")
    return knn, svm, rdf, xgb, gbc, lstm

# Load data (use st.cache_data for lightweight caching of the dataset)
@st.cache_data
def load_data():
    return pd.read_csv("/content/engine_data.csv")

# Prepare the prediction function
def predict_engine_health(input_data, model, scaler):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    if isinstance(model, tf.keras.Model):  # Check if the model is a Keras model (including LSTM)
        input_scaled = input_scaled.reshape((1, 1, input_scaled.shape[1]))
        prediction = model.predict(input_scaled)
        prediction = (prediction > 0.5).astype("int32").flatten()[0]
    else:
        prediction = model.predict(input_scaled)[0]
    return "Engine is in good health" if prediction == 0 else "Engine is not in good health"

# Main app
def main():
    st.title("Engine Health Prediction App")

    # Load data and models
    df = load_data()
    knn, svm, rdf, xgb, gbc, lstm = load_model_files()

    # Sidebar for user input
    st.sidebar.header("Input Engine Parameters")
    engine_rpm = st.sidebar.number_input("Enter Engine rpm", min_value=0.0, step=0.1)
    lub_oil_pressure = st.sidebar.number_input("Enter Lub Oil Pressure", min_value=0.0, step=0.1)
    fuel_pressure = st.sidebar.number_input("Enter Fuel Pressure", min_value=0.0, step=0.1)
    coolant_pressure = st.sidebar.number_input("Enter Coolant Pressure", min_value=0.0, step=0.1)
    lub_oil_temp = st.sidebar.number_input("Enter Lub Oil Temperature", min_value=0.0, step=0.1)
    coolant_temp = st.sidebar.number_input("Enter Coolant Temperature", min_value=0.0, step=0.1)

    input_data = [engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp]

    # Model selection
    model_choice = st.sidebar.selectbox("Choose a model", ['KNN', 'SVM', 'Random Forest', 'XGBoost', 'LSTM', 'Gradient Boosting'])

    # Load appropriate model
    if model_choice == "KNN":
        model = knn
    elif model_choice == "SVM":
        model = svm
    elif model_choice == "Random Forest":
        model = rdf
    elif model_choice == "XGBoost":
        model = xgb
    elif model_choice == "LSTM":
        model = lstm
    elif model_choice == "Gradient Boosting":
        model = gbc

    # Scale the input
    scaler = StandardScaler()
    scaler.fit(df.drop(columns=["Engine Condition"]))  # assuming "Engine Condition" is the target column

    if st.button("Predict"):
        result = predict_engine_health(input_data, model, scaler)
        st.write(f"Prediction: {result}")

    # Display Dataset
    if st.checkbox("Show Dataset"):
        st.write(df.head())

    # Plot correlation heatmap
    if st.checkbox("Show Correlation Heatmap"):
        correlation = df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
        st.pyplot(fig)

if __name__ == '__main__':
    main()
