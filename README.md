Potato Leaf Disease Prediction

Overview

This project is a web application for predicting potato leaf diseases using a machine learning model. The application is built using Flask for the backend and HTML, CSS, and PyScript for the frontend. Users can upload an image of a potato leaf, and the model will classify it as either healthy or diseased.

Features

Upload potato leaf images for disease detection

AI model for classifying leaf conditions

Interactive frontend using PyScript

Flask-based backend for handling requests

User-friendly interface

Technologies Used

Python

Flask

Machine Learning (CNN Model trained on a dataset)

Installation

Prerequisites

Python 3.7+

Flask

OpenCV

TensorFlow/Keras

NumPy


Steps to Run

Clone the repository:

git clone https://github.com/your-pjahanavi/potato-leaf-disease-prediction.git
cd potato-leaf-disease-prediction

Install dependencies:

pip install -r requirements.txt

Run the Flask application:

python app.py

Open a browser and go to:

http://127.0.0.1:5000

Dataset

The model is trained using a dataset of potato leaf images containing both healthy and diseased samples. The dataset was preprocessed and augmented to improve accuracy.

Model

A Convolutional Neural Network (CNN) model was used for classification.

The model was trained using TensorFlow/Keras.

The accuracy and loss metrics were monitored to ensure optimal performance.

Deployment

This web app can be deployed on platforms like:

Heroku

Render

AWS EC2

Google Cloud App Engine

Usage

Upload an image of a potato leaf.

Click the Predict button.

The model will analyze the image and display the prediction.

Future Improvements

Adding more disease categories.

Improving model accuracy with a larger dataset.

Implementing real-time mobile app integration.

Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.

License

This project is licensed under the MIT License. See LICENSE for details.

Contact

For queries, reach out at pjahanavi2811@gmail.com or visit GitHub Profile.

