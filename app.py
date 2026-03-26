import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# 1. Setup Data & Model (Keep your logic the same)
data = {
    "fever": [1,1,0,0,1,0],
    "cough": [1,1,0,0,0,0],
    "headache": [1,0,1,0,1,0],
    "fatigue": [1,1,0,1,1,0],
    "nausea": [0,0,1,0,1,0],
    "disease": ["Malaria","Flu","Migraine","Stress","Food Poisoning","Healthy"]
}

df = pd.DataFrame(data)
X = df[["fever","cough","headache","fatigue","nausea"]]
y = df["disease"]

model = DecisionTreeClassifier()
model.fit(X, y)

services = {
    "Malaria": "Visit a malaria treatment center.",
    "Flu": "Consult a general physician.",
    "Migraine": "Consult a neurologist.",
    "Stress": "Visit a mental health counselor.",
    "Food Poisoning": "Visit a hospital or clinic.",
    "Healthy": "No major illness detected."
}

# 2. Streamlit UI Layout
st.set_page_config(page_title="AI Health Checker", page_icon="🩺")

st.title("🩺 AI Health Symptom Checker")
st.write("Select your symptoms below to see a predicted condition. *Note: This is for educational purposes only.*")

st.divider()

# 3. Create Inputs using Columns
col1, col2 = st.columns(2)

with col1:
    fever = st.checkbox("Fever")
    cough = st.checkbox("Cough")
    headache = st.checkbox("Headache")

with col2:
    fatigue = st.checkbox("Fatigue")
    nausea = st.checkbox("Nausea")

# Convert Boolean (True/False) to Integer (1/0) for the model
input_data = [[int(fever), int(cough), int(headache), int(fatigue), int(nausea)]]

# 4. Prediction Logic
if st.button("Analyze Symptoms", type="primary"):
    prediction = model.predict(input_data)[0]
    
    st.subheader(f"Possible Condition: **{prediction}**")
    
    if prediction == "Healthy":
        st.success(services[prediction])
    else:
        st.warning(f"**Recommendation:** {services[prediction]}")
        
    # Extra: Show Confidence (optional)
    probs = model.predict_proba(input_data)
    st.info(f"Model Confidence: {max(probs[0]) * 100:.0f}%")