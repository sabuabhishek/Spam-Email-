import streamlit as st
import pickle
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="SPAM EMAIL CLASSIFIER",
    page_icon="📧",
    layout="centered"
)

# Load the model and vectorizer
@st.cache_resource
def load_model():
    model_path = r'C:\Users\ACER\Desktop\Spam or ham prjct\model.pkl'
    vectorizer_path = r'C:\Users\ACER\Desktop\Spam or ham prjct\vectorizer.pkl'
    
    # Debug information
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(vectorizer_path):
        st.error(f"Vectorizer file not found at {vectorizer_path}")
        raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def predict_spam(message, model, vectorizer):
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)[0]
    return 'SPAM' if prediction == 1 else 'HAM'

def main():
    st.title(" SPAM EMAIL CLASSIFIER")
    st.write("Enter a message to check if it's spam or ham")

    # Load model
    try:
        model, vectorizer = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return 

    # Text input
    message = st.text_area("Enter your message:", height=150)

    if st.button("Predict"):
        if message.strip() != "":
            # Make prediction
            result = predict_spam(message, model, vectorizer)
            
            # Display result with color coding
            if result == "SPAM":
                st.error(f"Prediction: {result}")
            else:
                st.success(f"Prediction: {result}")
        else:
            st.warning("Please enter a message!")

    # Add information section
    #with st.expander("ℹ️ About"):
     #   st.write("""
      #  This app uses a machine learning model to predict whether a message is spam or ham (not spam).
       # - Spam: Unwanted, unsolicited messages
        #- Ham: Normal, legitimate messages
        #""")

if __name__ == "__main__":
    main()
