import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="HR Promotion Predictor",
    page_icon="📈",
    layout="wide"
)

# -------------------------------
# Load Model and Feature Columns
# -------------------------------
with open("hr_promotion_pipeline.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
feature_columns = artifacts["feature_columns"]

# -------------------------------
# Title and Description
# -------------------------------
st.title("📈 HR Employee Promotion Prediction")
st.markdown(
    """
    Predict whether an employee is likely to get promoted based on
    performance, education, department, and other HR attributes.
    """
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("ℹ️ About Project")
st.sidebar.info(
    """
    This application uses a Machine Learning model trained on HR analytics data
    to predict employee promotion eligibility.
    """
)

# -------------------------------
# Input Layout
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    department = st.selectbox(
        "Department",
        [
            "Sales & Marketing",
            "Operations",
            "Technology",
            "Analytics",
            "Finance",
            "HR",
            "Procurement",
            "Legal",
            "R&D"
        ]
    )

    education = st.selectbox(
        "Education",
        [
            "Bachelor's",
            "Master's & above",
            "Below Secondary"
        ]
    )

    gender = st.selectbox(
        "Gender",
        ["m", "f"]
    )

    recruitment_channel = st.selectbox(
        "Recruitment Channel",
        ["sourcing", "referred", "other"]
    )

    no_of_trainings = st.number_input(
        "Number of Trainings",
        min_value=1,
        max_value=20,
        value=1
    )

with col2:
    age = st.number_input(
        "Age",
        min_value=20,
        max_value=60,
        value=30
    )

    previous_year_rating = st.slider(
        "Previous Year Rating",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=1.0
    )

    length_of_service = st.number_input(
        "Length of Service (Years)",
        min_value=1,
        max_value=40,
        value=5
    )

    awards_won = st.selectbox(
        "Awards Won?",
        [0, 1]
    )

    avg_training_score = st.slider(
        "Average Training Score",
        min_value=30,
        max_value=100,
        value=60
    )

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🚀 Predict Promotion", use_container_width=True):

    # Create input dictionary
    input_data = {
        "department": department,
        "education": education,
        "gender": gender,
        "recruitment_channel": recruitment_channel,
        "no_of_trainings": no_of_trainings,
        "age": age,
        "previous_year_rating": previous_year_rating,
        "length_of_service": length_of_service,
        "awards_won?": awards_won,
        "avg_training_score": avg_training_score
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # One-Hot Encoding
    df = pd.get_dummies(
        df,
        columns=["department", "recruitment_channel"]
    )

    # Label Encoding
    education_map = {
        "Below Secondary": 0,
        "Bachelor's": 1,
        "Master's & above": 2
    }

    gender_map = {
        "f": 0,
        "m": 1
    }

    df["education"] = df["education"].map(education_map)
    df["gender"] = df["gender"].map(gender_map)

    # Align columns with training data
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Prediction
    prediction = model.predict(df)[0]

    # Probability (if supported)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(df)[0][1]
    else:
        probability = 0.5

    # -------------------------------
    # Results
    # -------------------------------
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success("🎉 Employee is likely to be PROMOTED.")
    else:
        st.error("❌ Employee is NOT likely to be promoted.")

    st.write(f"### Confidence Score: {probability:.2%}")
    st.progress(float(probability))

    # Input Summary
    st.subheader("📋 Input Summary")
    st.dataframe(pd.DataFrame([input_data]), use_container_width=True)

    # Download Result
    result_df = pd.DataFrame({
        "Prediction": ["Promoted" if prediction == 1 else "Not Promoted"],
        "Confidence": [f"{probability:.2%}"]
    })

    st.download_button(
        label="📥 Download Result",
        data=result_df.to_csv(index=False),
        file_name="promotion_prediction.csv",
        mime="text/csv"
    )