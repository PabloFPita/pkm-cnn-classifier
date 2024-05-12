
<img src="img/pokemon-logo.png" width="300">

# Pokemon Classifier (1st Generation only)

## Description
This project is a Pokemon classifier that can identify the first generation of Pokemon. The model is trained using transfer learning with a pre-trained ResNet152 model. The dataset used is the **Pokemon-151 dataset** from Kaggle (https://www.kaggle.com/datasets/mikoajkolman/pokemon-images-first-generation17000-files). The model is trained on Google Colab and the best model is saved to be used in the Streamlit web application. The web application allows the user to upload an image of a Pokemon and the model will predict the Pokemon's name.

Experiment tracking is done with **Weights and Biases** to monitor the performance of the models and to refine them.

## Installation
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

## Usage
To **run the Streamlit web application**, execute the following command:
```bash
streamlit run app.py
```

To **train different models**, modify the transfer-learning-training.ipynb notebook and execute it. Make sure to have a Weights and Biases account to log the experiments and add the **API key** to a .env file.

## The model
The weights of the best model are saved in the **models** folder. The best current model is a ResNet152 model with a final fully connected layer with 143 output units (one for each Pokemon). The model is trained with a learning rate of 0.0001 and a batch size of 32. The model is trained for 10 epochs.

**Note**:
This weights are used in the Streamlit web application to predict the Pokemon's name. But, as of now, it is not versioned on git due to its size.