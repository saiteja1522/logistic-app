import streamlit as st
import pandas as pd
import joblib
import pickle

# Load the trained model
try:
    with open('model_pickle', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please upload 'model_pickle' to the app directory.")
    st.stop()


st.title("Titanic Survival Prediction")

# Create input fields for user data
pclass = st.slider("Passenger Class", 1, 3, 1)
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, value=0)
fare = st.number_input("Fare", min_value=0.0, value=10.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])
name = st.text_input("Name", "Enter name here")


# Create a button to trigger prediction
if st.button("Predict"):
    # Create a DataFrame with the user input
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked],
        'Name': [name]
    })

    # Perform preprocessing on the input data (similar to how you preprocessed the training data)
    # ... (Add preprocessing steps for numerical values similar to the original script) ...
    # Example mapping, replace with your actual mapping
    sex_mapping = {"male": 0, "female": 1}
    embarked_mapping = {"C": 0, "Q": 1, "S": 2}
    input_data["Sex"] = input_data["Sex"].map(sex_mapping)
    input_data["Embarked"] = input_data["Embarked"].map(embarked_mapping)

    try:
      # Make prediction using the loaded model
      prediction = model.predict(input_data)[0]
      
      if prediction == 0:
          st.write("Prediction: Did not survive")
      else:
          st.write("Prediction: Survived")

    except Exception as e:
      st.error(f"Error during prediction: {e}")
