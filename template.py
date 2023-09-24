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
        st.title("Single Prediction")
        st.write("Enter the necessary input and see live predictions.")

        # Add your input fields here
        # For example:
        input_text = st.text_input("Enter text for prediction")

        # Add a button to trigger the prediction
        if st.button("Predict"):
            # Call your model and perform the prediction here
            # For example:
            prediction = _singlePredict(input_text)

            # Display the prediction result
            st.subheader(f"Prediction: {prediction}")

# Define your model prediction function here
# For example:

# We are going to use st.cache to improve performance for predictions.
@st.cache_data
def _singlePredict(input_text):
    # Format the input_text so that you can pass it to the model
    # For example:
    # Call your model to make predictions on the input_text
    # For example:
    prediction = model.predict([[float(input_text)]])

    # Make sure to return the prediction result
    return prediction[0][0]

# Run the app
if __name__ == "__main__":
    main()
