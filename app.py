# import streamlit as st
# import pandas as pd
# import joblib

# st.set_page_config(page_title="Student Health Assessment", layout="wide")

# @st.cache_resource
# def load_artifacts():
#     preprocessor = joblib.load('preprocessor.pkl')
#     label_encoder = joblib.load('label_encoder.pkl')
#     model = joblib.load('best_health_model.pkl')
#     return preprocessor, label_encoder, model

# try:
#     preprocessor, label_encoder, model = load_artifacts()
# except Exception as e:
#     st.error(f"Error loading artifacts: {e}. Run training script first.")
#     st.stop()

# st.title("Student Health Assessment Portal")
# st.write("Enter student health metrics below to generate a health status assessment.")

# st.markdown("---")

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.subheader("Physical Metrics")
#     sleep_duration = st.number_input("Sleep Duration (hrs)", 0.0, 24.0, 7.0, 0.1)
#     heart_rate = st.number_input("Heart Rate (bpm)", 40.0, 180.0, 72.0, 1.0)
#     bmi = st.number_input("BMI", 10.0, 50.0, 22.0, 0.1)
#     calorie_expenditure = st.number_input("Calorie Expenditure (kcal)", 500.0, 5000.0, 2000.0, 10.0)
#     step_count = st.number_input("Step Count", 0, 40000, 8000, 500)

# with col2:
#     st.subheader("Daily Activities")
#     exercise_duration = st.number_input("Exercise Duration (mins)", 0.0, 300.0, 30.0, 5.0)
#     water_intake = st.number_input("Water Intake (Liters)", 0.0, 10.0, 2.5, 0.1)
#     screen_time = st.number_input("Screen Time (hrs)", 0.0, 24.0, 6.0, 0.5)
#     sitting_time = st.number_input("Sitting Time (hrs)", 0.0, 24.0, 8.0, 0.5)
#     gender = st.selectbox("Gender", ["male", "female", "other"])

# with col3:
#     st.subheader("Behavioral and Social Metrics")
#     diet_type = st.selectbox("Diet Type", ["balanced", "veg", "non-veg"])
#     stress_level = st.selectbox("Stress Level", ["low", "medium", "high"])
#     sleep_quality = st.selectbox("Sleep Quality", ["poor", "average", "good"])
#     physical_activity_level = st.selectbox("Physical Activity", ["sedentary", "moderate", "active"])
#     smoking_status = st.selectbox("Smoking Status", ["no", "occasional", "yes"])
#     alcohol_intake = st.selectbox("Alcohol Intake", ["no", "moderate", "high"])
#     mental_health_status = st.selectbox("Mental Health Status", ["poor", "moderate", "stable"])
#     academic_pressure = st.selectbox("Academic Pressure", ["low", "medium", "high"])
#     social_relationships = st.selectbox("Social Relationships", ["poor", "average", "good"])

# if st.button("Generate Health Assessment", use_container_width=True):
#     input_data = pd.DataFrame([{
#         'sleep_duration': sleep_duration,
#         'heart_rate': heart_rate,
#         'bmi': bmi,
#         'calorie_expenditure': calorie_expenditure,
#         'step_count': step_count,
#         'exercise_duration': exercise_duration,
#         'water_intake': water_intake,
#         'screen_time': screen_time,
#         'sitting_time': sitting_time,
#         'diet_type': diet_type,
#         'stress_level': stress_level,
#         'sleep_quality': sleep_quality,
#         'physical_activity_level': physical_activity_level,
#         'smoking_status': smoking_status,
#         'alcohol_intake': alcohol_intake,
#         'mental_health_status': mental_health_status,
#         'academic_pressure': academic_pressure,
#         'social_relationships': social_relationships,
#         'gender': gender
#     }])
    
#     # Process inputs and model prediction
#     prep_data = preprocessor.transform(input_data)
#     pred_idx = model.predict(prep_data)[0]
#     pred_label = label_encoder.inverse_transform([pred_idx])[0]
#     pred_proba = model.predict_proba(prep_data)[0]

#     st.markdown("---")
#     st.subheader("Health Assessment Result")

#     # Single status metric
#     col_status, col_info = st.columns([1, 2])
    
#     with col_status:
#         if pred_label == 'fit':
#             st.metric(label="Overall Status", value="FIT")
#             st.success("Student is in optimal health condition.")
#         elif pred_label == 'at-risk':
#             st.metric(label="Overall Status", value="AT-RISK")
#             st.warning("Elevated health risk detected.")
#         else:
#             st.metric(label="Overall Status", value="UNHEALTHY")
#             st.error("Critical health concerns detected.")

#     with col_info:
#         st.markdown("**Assessment Overview**")
#         if pred_label == 'fit':
#             st.write("Daily habits, physical parameters, and mental health metrics are well-balanced.")
#         elif pred_label == 'at-risk':
#             st.write("Current daily habits place the student in an elevated risk category. Proactive lifestyle adjustments are recommended.")
#         else:
#             st.write("Multiple metrics show critical health risks requiring direct medical or lifestyle intervention.")

#     # Practical key factors to address
#     st.subheader("Actionable Observations")
#     concerns = []
#     if sleep_duration < 7.0:
#         concerns.append(f"Low Sleep Duration: {sleep_duration} hours recorded (Recommended: 7.0 to 9.0 hours).")
#     if screen_time > 6.0:
#         concerns.append(f"High Screen Time: {screen_time} hours recorded per day.")
#     if exercise_duration < 30.0:
#         concerns.append(f"Low Exercise Duration: {exercise_duration} minutes recorded per day.")
#     if stress_level in ['medium', 'high']:
#         concerns.append(f"Stress Level: Currently recorded as {stress_level.capitalize()}.")
#     if water_intake < 2.0:
#         concerns.append(f"Low Water Intake: {water_intake} Liters recorded per day.")

#     if concerns:
#         for item in concerns:
#             st.write(f"- {item}")
#     else:
#         st.write("- All major habits appear within healthy target ranges.")

#     # Technical hidden section
#     with st.expander("Technical Model Details (For Reference Only)"):
#         prob_pairs = sorted(zip(label_encoder.classes_, pred_proba), key=lambda x: x[1], reverse=True)
#         for cls_name, prob_val in prob_pairs:
#             st.write(f"{cls_name.capitalize()}: {prob_val * 100:.1f}% confidence")



import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Health Predictor", layout="wide")

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load('preprocessor.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    model = joblib.load('best_health_model.pkl')
    return preprocessor, label_encoder, model

try:
    preprocessor, label_encoder, model = load_artifacts()
except Exception as e:
    st.error(f"Error loading artifacts: {e}. Ensure training script/notebook has run and generated .pkl files.")
    st.stop()

st.title("Student Health Condition Prediction System")
st.write("Input lifestyle, mental, and physical health attributes to test student status.")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Physical Metrics")
    sleep_duration = st.number_input("Sleep Duration (hrs)", 0.0, 24.0, 7.0, 0.1)
    heart_rate = st.number_input("Heart Rate (bpm)", 40.0, 180.0, 72.0, 1.0)
    bmi = st.number_input("BMI", 10.0, 50.0, 22.0, 0.1)
    calorie_expenditure = st.number_input("Calorie Expenditure (kcal)", 500.0, 5000.0, 2000.0, 10.0)
    step_count = st.number_input("Step Count", 0, 40000, 8000, 500)

with col2:
    st.subheader("Daily Activities")
    exercise_duration = st.number_input("Exercise Duration (mins)", 0.0, 300.0, 30.0, 5.0)
    water_intake = st.number_input("Water Intake (Liters)", 0.0, 10.0, 2.5, 0.1)
    screen_time = st.number_input("Screen Time (hrs)", 0.0, 24.0, 6.0, 0.5)
    sitting_time = st.number_input("Sitting Time (hrs)", 0.0, 24.0, 8.0, 0.5)
    gender = st.selectbox("Gender", ["male", "female", "other"])

with col3:
    st.subheader("Behavioral and Social Metrics")
    diet_type = st.selectbox("Diet Type", ["balanced", "veg", "non-veg"])
    stress_level = st.selectbox("Stress Level", ["low", "medium", "high"])
    sleep_quality = st.selectbox("Sleep Quality", ["poor", "average", "good"])
    physical_activity_level = st.selectbox("Physical Activity", ["sedentary", "moderate", "active"])
    smoking_status = st.selectbox("Smoking Status", ["no", "occasional", "yes"])
    alcohol_intake = st.selectbox("Alcohol Intake", ["no", "moderate", "high"])
    mental_health_status = st.selectbox("Mental Health Status", ["poor", "moderate", "stable"])
    academic_pressure = st.selectbox("Academic Pressure", ["low", "medium", "high"])
    social_relationships = st.selectbox("Social Relationships", ["poor", "average", "good"])

if st.button("Predict Health Condition", use_container_width=True):
    input_data = pd.DataFrame([{
        'sleep_duration': sleep_duration,
        'heart_rate': heart_rate,
        'bmi': bmi,
        'calorie_expenditure': calorie_expenditure,
        'step_count': step_count,
        'exercise_duration': exercise_duration,
        'water_intake': water_intake,
        'screen_time': screen_time,
        'sitting_time': sitting_time,
        'diet_type': diet_type,
        'stress_level': stress_level,
        'sleep_quality': sleep_quality,
        'physical_activity_level': physical_activity_level,
        'smoking_status': smoking_status,
        'alcohol_intake': alcohol_intake,
        'mental_health_status': mental_health_status,
        'academic_pressure': academic_pressure,
        'social_relationships': social_relationships,
        'gender': gender
    }])
    
    # Process inputs and predict
    prep_data = preprocessor.transform(input_data)
    pred_idx = model.predict(prep_data)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    pred_proba = model.predict_proba(prep_data)[0]

    st.markdown("---")
    st.subheader("Health Assessment Result")

    # Clear single status metric display
    col_status, col_info = st.columns([1, 2])
    
    with col_status:
        if pred_label == 'fit':
            st.metric(label="Overall Status", value="FIT")
            st.success("Student is in optimal health condition.")
        elif pred_label == 'at-risk':
            st.metric(label="Overall Status", value="AT-RISK")
            st.warning("Early indicators show elevated health or lifestyle risks.")
        else:
            st.metric(label="Overall Status", value="UNHEALTHY")
            st.error("Multiple metrics indicate critical health concerns.")

    with col_info:
        st.markdown("**Assessment Overview**")
        if pred_label == 'fit':
            st.write("Physical activity, sleep quality, and daily habits meet recommended guidelines.")
        elif pred_label == 'at-risk':
            st.write("Current daily habits place the student in an elevated risk category. Proactive lifestyle adjustments are recommended.")
        else:
            st.write("Habits and physical indicators show substantial health concerns requiring attention and lifestyle intervention.")

    # Identified factors needing improvement
    st.subheader("Key Factors to Focus On")
    
    concerns = []
    if sleep_duration < 7.0:
        concerns.append(f"Low Sleep Duration: Recorded {sleep_duration} hrs (Recommended: 7.0-9.0 hrs).")
    if screen_time > 6.0:
        concerns.append(f"High Screen Time: Recorded {screen_time} hrs/day.")
    if exercise_duration < 30.0:
        concerns.append(f"Low Daily Exercise: Recorded {exercise_duration} mins/day.")
    if stress_level in ['medium', 'high']:
        concerns.append(f"Stress Level: Currently marked as {stress_level.capitalize()}.")
    if water_intake < 2.0:
        concerns.append(f"Low Water Intake: Recorded {water_intake} L/day (Recommended: 2.5+ L).")

    if concerns:
        for item in concerns:
            st.write(f"- {item}")
    else:
        st.write("- All major habits appear within healthy ranges.")

    # Hidden probability breakdown inside expander
    with st.expander("Technical Model Details (Optional)"):
        prob_pairs = sorted(zip(label_encoder.classes_, pred_proba), key=lambda x: x[1], reverse=True)
        for cls_name, prob_val in prob_pairs:
            st.write(f"**{cls_name.capitalize()}**: {prob_val * 100:.1f}% probability")