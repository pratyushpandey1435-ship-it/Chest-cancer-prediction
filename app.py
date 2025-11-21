import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path
import tempfile
from constants import *
from utils import preprocess_dcm

# Load the trained models
@st.cache_resource
def load_models():
    rf_model = joblib.load(RF_MODEL)
    cnn_model = tf.keras.models.load_model(CNN_MODEL)
    return rf_model, cnn_model

def main():
    st.title("Heart Failure Prediction System")
    
    # Load models
    rf_model, cnn_model = load_models()
    
    # Create tabs for different prediction methods
    tab1, tab2 = st.tabs(["Clinical Data Prediction", "MRI DICOM Image Prediction"])
    
    # Tab 1: Clinical Data Prediction (Random Forest)
    with tab1:
        st.header("Enter Clinical Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=100, value=60)
            anaemia = st.selectbox("Anaemia", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", min_value=0, value=581)
            diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100, value=38)
            high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            
        with col2:
            platelets = st.number_input("Platelets", min_value=0, value=265000)
            serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0, value=1.9)
            serum_sodium = st.number_input("Serum Sodium", min_value=0, value=130)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
            smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            time = st.number_input("Follow-up Period (days)", min_value=0, value=4)
        
        if st.button("Predict using Clinical Data"):
            # Create input dataframe
            input_data = pd.DataFrame({
                'age': [age],
                'anaemia': [anaemia],
                'creatinine_phosphokinase': [creatinine_phosphokinase],
                'diabetes': [diabetes],
                'ejection_fraction': [ejection_fraction],
                'high_blood_pressure': [high_blood_pressure],
                'platelets': [platelets],
                'serum_creatinine': [serum_creatinine],
                'serum_sodium': [serum_sodium],
                'sex': [sex],
                'smoking': [smoking],
                'time': [time]
            })
            
            # Make prediction
            prediction = rf_model.predict_proba(input_data)[0]
            
            # Display results
            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Survival Probability", f"{(1 - prediction[1]):.2%}")
            with col2:
                st.metric("Heart Failure Risk", f"{prediction[1]:.2%}")
            
            # Display risk level
            risk_level = "High" if prediction[1] > 0.7 else "Moderate" if prediction[1] > 0.3 else "Low"
            st.info(f"Risk Level: {risk_level}")
    
    # Tab 2: MRI Image Prediction (CNN)
    with tab2:
        st.header("Upload MRI Image")
        uploaded_file = st.file_uploader("Choose a DICOM (.dcm) file", type=['dcm'])
        
        if uploaded_file is not None:
            # Create a temporary file to save the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Process the image
                image = preprocess_dcm(tmp_file_path)
                
                # Display the processed image
                st.image(image, caption='Uploaded MRI Image', use_column_width=True)
                
                # Prepare image for prediction
                image_batch = np.expand_dims(image, axis=0)
                
                # Make prediction
                prediction = cnn_model.predict(image_batch)[0]
                
                # Display results
                st.subheader("Prediction Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Normal Probability", f"{prediction[0]:.2%}")
                with col2:
                    st.metric("Heart Failure Probability", f"{prediction[1]:.2%}")
                
                # Display prediction class
                predicted_class = "Heart Failure Detected" if prediction[1] > 0.5 else "Normal"
                confidence = max(prediction[0], prediction[1])
                st.info(f"Prediction: {predicted_class} (Confidence: {confidence:.2%})")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
            
            finally:
                # Clean up temporary file
                Path(tmp_file_path).unlink()

if __name__=="__main__":
    main()