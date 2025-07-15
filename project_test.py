import streamlit as st
import pandas as pd
import joblib

model_path = "social_media_addiction_model.pkl"
model = joblib.load(model_path)

# Input form
st.title("Social Media Addiction Score Predictor")
st.write("Enter a student's data below to predict their social media addiction score.")

# Input fields
age = st.slider("Age", 15, 35, 21)
gender = st.selectbox("Gender", ['Male', 'Female'])
avg_daily_usage_hours = st.slider("Average Daily Social Media Use (hours)", 0.0, 24.0, 4.0)
sleep_hours_per_night = st.slider("Sleep Hours per Night", 0.0, 12.0, 7.0)
relationship_status = st.selectbox("Relationship Status", ['Single', 'In a Relationship', 'Married'])
most_used_platform = st.selectbox("Most Used Platform", ['Instagram', 'Facebook', 'Snapchat', 'TikTok', 'WhatsApp', 'Other'])

# Process inputs into a DataFrame
def preprocess_input():
    data = {
        'age': [age],
        'avg_daily_usage_hours': [avg_daily_usage_hours],
        'sleep_hours_per_night': [sleep_hours_per_night],
        'gender_Male': [1 if gender == 'Male' else 0],
        'relationship_status_In a Relationship': [1 if relationship_status == 'In a Relationship' else 0],
        'relationship_status_Married': [1 if relationship_status == 'Married' else 0],
        'most_used_platform_Facebook': [1 if most_used_platform == 'Facebook' else 0],
        'most_used_platform_Instagram': [1 if most_used_platform == 'Instagram' else 0],
        'most_used_platform_Snapchat': [1 if most_used_platform == 'Snapchat' else 0],
        'most_used_platform_TikTok': [1 if most_used_platform == 'TikTok' else 0],
        'most_used_platform_WhatsApp': [1 if most_used_platform == 'WhatsApp' else 0],
        # 'Other' is baseline (drop_first used)
    }

    # Add missing columns as 0 to match training columns if needed
    full_feature_set = model.feature_names_in_
    df_input = pd.DataFrame(data)
    for col in full_feature_set:
        if col not in df_input.columns:
            df_input[col] = 0

    return df_input[full_feature_set]

if st.button("Predict Addiction Score"):
    input_df = preprocess_input()
    prediction = model.predict(input_df)[0]

    # Determine color and message based on prediction
    if prediction < 8.6:
        color = "green"
        st.info("🟢 Low risk: Healthy social media habits.")
    elif 8.6 <= prediction < 9.5:
        color = "goldenrod"
        st.warning("🟡 Moderate risk: Some signs of overuse. Consider reducing screen time.")
    elif 9.5 <= prediction < 9.7:
        color = "orange"
        st.warning("🟠 High risk: Noticeable impact from social media usage.")
    else:
        color = "red"
        st.error("🔴 Very High risk: Possible addiction. Strongly consider lifestyle changes.")

    # Display the styled score
    st.markdown(
        f"""
        <h2 style='text-align: center;'>
            Addiction Score: <strong style='color: {color};'>{prediction:.2f}</strong>
        </h2>
        """,
        unsafe_allow_html=True
    )