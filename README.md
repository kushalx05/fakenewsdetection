# fakenewsdetection

# Fake News Prediction Project

This project aims to build a machine learning model that can accurately classify news articles as either true or fake. 

## Project Overview

The project involves the following steps:

1. **Data Collection:** Gathering a dataset of true and fake news articles.
2. **Data Preprocessing:** Cleaning and preparing the data for model training. This includes tasks like removing punctuation, stop words, and converting text to lowercase.
3. **Feature Engineering:** Extracting relevant features from the text data, such as using TF-IDF to represent the text numerically.
4. **Model Training:** Training a machine learning model, such as a feedforward neural network (FFNN), on the preprocessed data.
5. **Model Evaluation:** Evaluating the model's performance on a separate test dataset using metrics like accuracy, precision, recall, and F1-score.
6. **Deployment (Optional):** Creating an interface for users to input news articles and receive predictions from the model.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* PyTorch
* Jupyter Notebook (Google Colab)

## Usage

1. **Data:** Place the `True.csv` and `Fake.csv` files in the project directory.
2. **Colab:** Open the Jupyter Notebook in Google Colab.
3. **Run:** Execute the notebook cells to load the data, preprocess it, train the model, and evaluate its performance.
4. **Prediction:** Input a news article using the provided input prompt to get a prediction from the model.

## Results

The trained model achieved an accuracy of 98.98 on the test dataset. Other evaluation metrics are included in the notebook.

## Future Work

* Experiment with different model architectures and hyperparameters.
* Explore more advanced feature engineering techniques.
* Incorporate external knowledge sources, such as fact-checking websites.
* Deploy the model as a web application for real-time predictions.
