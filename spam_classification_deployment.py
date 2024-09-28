import streamlit as st
import pickle
from PIL import Image
import os

# Set the page title
st.set_page_config(page_title="Spam E-Mail Classification")

# Use color and font themes
st.markdown("""
<style>
div[class*="stTextInput"] label p {
  font-size: 26px;
}
</style>
""", unsafe_allow_html=True)

# Load the model and vectorizer using relative paths
pickle_dir = os.path.join(os.getcwd(), 'Pickle Files')
model_path = os.path.join(pickle_dir, 'model.pkl')
feature_path = os.path.join(pickle_dir, 'feature.pkl')

tfidf = pickle.load(open(feature_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))

st.title("Spam E-Mail Classifier")

# Load the image using a relative path
image_path = os.path.join(os.getcwd(), 'Data Source', 'images.jpg')
image = Image.open(image_path)
st.image(image, use_column_width=True)

# Input for the email message
input_mail = st.text_input("Enter the Message")

# Prediction button
if st.button('Predict'):
    # Transform the input using the TF-IDF vectorizer
    vector_input = tfidf.transform([input_mail])
    
    # Predict using the trained model
    result = model.predict(vector_input)
    
    # Display the result
    st.success("This is a " + ('Spam Mail' if result == 1 else 'Ham Mail'))
