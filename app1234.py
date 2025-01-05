import streamlit as st
import pickle
import numpy as np

# Load the new trained model (logistic_regression_model1.pkl)
with open('logistic_regression_model1.pkl', 'rb') as file:
    model = pickle.load(file)

# App Title
st.title("Titanic Survival Prediction: Logistic Regression Model 1")

# User Inputs
st.header("Input Passenger Features")

# Collect inputs for each feature
pclass = st.selectbox("Passenger Class (Pclass)", options=[1, 2, 3], format_func=lambda x: f"Class {x}")
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
age = st.number_input("Age (in years)", min_value=0.0, step=1.0)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, step=1)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, step=1)
fare = st.number_input("Passenger Fare (Fare)", min_value=0.0, step=0.01)
embarked = st.selectbox("Port of Embarkation (Embarked)", options=[0, 1, 2], format_func=lambda x: ["C (Cherbourg)", "Q (Queenstown)", "S (Southampton)"][x])

# Prediction
if st.button("Predict"):
    # Arrange input features in the expected order
    input_features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    
    # Predict using the new trained model
    prediction = model.predict(input_features)
    
    # Display result
    st.write("Prediction:", "Survived" if prediction[0] == 1 else "Did Not Survive")
