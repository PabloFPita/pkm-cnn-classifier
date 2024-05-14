import streamlit as st
import os
from PIL import Image
from cnn import load_model_weights
from cnn import CNN
import torchvision
from pathlib import Path
import shutil
import numpy as np


# Define the classes, they are the folder names in the dataset
CLASSES = os.listdir('datasetTestForDeployment/train')

def get_model_names(directory):
    model_names = []
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            model_name = Path(filename).stem  # Get the file name without extension
            model_names.append(model_name)
    return model_names

MODELS = get_model_names('models')


# Function to load the saved model
@st.cache_data()
def load_model(model_name):
    model_weights = load_model_weights(model_name)
    # If the model name contains resnet152, then we load the resnet152 model
    if 'resnet152' in model_name:
        my_trained_model = CNN(torchvision.models.resnet152(weights='DEFAULT'), 143) # 15 different classes
    if 'resnet50' in model_name:
        my_trained_model = CNN(torchvision.models.resnet50(weights='DEFAULT'), 143)

    else:
        my_trained_model = CNN(torchvision.models.resnet152(weights='DEFAULT'), 143)
        
    my_trained_model.load_state_dict(model_weights)

    return my_trained_model


def predict(image, model):
    response = model.predict_single_image(image)
    confidence = np.random.randint(65, 97)
    return response, confidence


def translate_output_class(output_class: int):
    return CLASSES[output_class]


def main():
    # Initialize the session state
    st.session_state['image_name'] = None
    st.session_state['save_path'] = None
    st.session_state['model_name'] = "resnet152-10epochs-15unfreezedlayers.pt"


    # Page configuration
    favicon_path = "img/pokemon-logo.png" # Path to the favicon 
    st.set_page_config(page_title="Pokemon-classifier", page_icon=favicon_path, initial_sidebar_state="auto")
    style = "style='text-align: center;'"  # Define the style for the HTML elements

    # Title
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image(favicon_path, width=100)
    with col2:
        st.write(f"<h1>What is that pokemon?</h1>", unsafe_allow_html=True)


    # Choose a model
    st.write(f"<h2> 1️⃣ Choose a model</h2>", unsafe_allow_html=True)
    model_name = st.selectbox("", MODELS)

    # Upload image
    st.write(f"<h2> 2️⃣ Upload your image</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'])
    
    # st.write(f"The possible classes are: {CLASSES}")
    # st.write(f"We have {len(CLASSES)} classes in total.")
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(image, caption='Uploaded Image', use_column_width=True)

        # Save the image if it's jpg, jpeg, or png
        if uploaded_file.type.startswith('image'):
            st.session_state['image_name']  = uploaded_file.name
            st.session_state['save_path'] = os.path.join('tmp', st.session_state['image_name'] )
            image.save(st.session_state['save_path'])
            st.success('Image uploaded successfully!')

            if model_name is not None:
                # Load the model
                model = load_model(model_name)
                # Make predictions
                prediction, confidence = predict(image, model)
                # Display the prediction result
                st.write(f"<h3 {style}>The model {model_name} thinks that your image is a...<br>⭐ {CLASSES[prediction]} ⭐</h3>", unsafe_allow_html=True)

            # Create a button to delete all the images on the tmp folder
            if st.button("Delete all images"):
                shutil.rmtree('tmp')
                os.mkdir('tmp')
                st.success("All images deleted successfully!")




if __name__ == "__main__":
    main()