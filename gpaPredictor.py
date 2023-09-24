import streamlit as st
import joblib
import pandas as pd
import os

# 1. Import your model and any necessary dependencies here
if os.path.exists("Model/gpaPredictor.joblib"):
    model = joblib.load("Model/gpaPredictor.joblib")


# 2. Set up your Streamlit app
def main():
    # (Optional) Set page title and favicon.
    st.set_page_config(page_title="First ML Project", page_icon="👀")

    # (Optional) Set a sidebar for your app.
    with st.sidebar:
        # st.image("IMAGE_PATH")
        st.title("SIDE_BAR_TITLE")
        choice = st.radio(
            "Menu", ["Home", "Single GPA Prediction"])
        st.info(
            "PROJECT_DESCRIPTION")
    
    # Now lets add content to each sub-page of your site
    if choice == "Home":
        # Add a title and some text to the app:
        st.title("Ultimate GPA Predictor")
        st.write("See what your writing score would be.")

    elif choice == "Single GPA Prediction":
        # Add a title and some text to the app:
        st.title("Batch Prediction")
        st.write("Upload a CSV file and see live predictions.")

        # Add a file uploader to upload a CSV file
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        # If a file is uploaded, process and display predictions
        if uploaded_file is not None:
            try:
              df = pd.read_csv(uploaded_file, sep=",")
            except Exception as e:
                st.error("Error: Invalid CSV file. Please upload a valid CSV file.")
            # Display the uploaded data
            st.subheader("Input Data")
            st.dataframe(df, use_container_width=True)

            # Perform predictions on the uploaded data
            predictions = _batchPredict(df)

            # Display the prediction results
            st.subheader("Prediction Results")
            st.dataframe(predictions, use_container_width=True)

# Define your model prediction function here
# For example:

# We are going to use st.cache to improve performance for predictions.
@st.cache_data
def _batchPredict(df):
    # Format the dataframe so that you can pass it to the model
    # For example:
    df = df[["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]]

    # Call your model to make predictions on the dataframe
    # For example:
    predictions = model.predict(df)

    # Predictions DF
    dfPredictions = pd.DataFrame(predictions, columns=(["Writing score"]))

    # Make sure to return the prediction results
    return dfPredictions

# Run the app
if __name__ == "__main__":
    main()
