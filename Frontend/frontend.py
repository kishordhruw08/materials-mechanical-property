import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000//predict" 

st.title("Material's Properties Predictor")
st.markdown("Enter your details below:")

# Input fields
formula = st.text_input("Chemical Formula", value="Al2O3")

if st.button("Predict Property"):
    input_data = {
        "formula":formula    
    }

    try:
        response = requests.post(API_URL, json=input_data)
        result = response.json()

        if response.status_code == 200 and "response" in result:
            prediction = result["response"]
            st.success(f"Predicted Material Properties: **{prediction['predicted_category']}**")

        else:
            # st.error(f"API Error: {response.status_code}")
            st.write(result)

    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the FastAPI server. Make sure it's running.")
