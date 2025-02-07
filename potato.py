import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Define model if you haven't trained it yet, or load an existing model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load a pre-trained model if available, otherwise create and train a new one
try:
    model = load_model('plant_disease_model.h5')
except:
    model = create_model()
    # Model training code goes here if you need to train it.
    # model.fit(...) 
    # Save the model after training
    # model.save('plant_disease_model.h5')

# Function to load and preprocess an image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1] range
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array
img_array = preprocess_image(r"C:\Users\jaanu\Desktop\potatod.jpg")
print(img_array.shape, img_array.min(), img_array.max())  # Check shape and value range


# Function to make a prediction
def predict_disease(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    if prediction[0] < 0.5:
        result = "Diseased"
    if prediction[0] > 0.5:
        result = "Healthy"
    
    # Display the image with the prediction
    img = image.load_img(img_path, target_size=(150, 150))
    plt.imshow(img)
    plt.title(f"Prediction: {result}")
    plt.axis('off')
    plt.show()
    
    return result

# Example usage:
# Replace 'path_to_leaf_image.jpg' with 
result = predict_disease(r"C:\Users\jaanu\Desktop\potatod.jpg")
print(f"The plant is: {result}")
prediction = model.predict(img_array)
print(f"Raw prediction value: {prediction[0][0]}")

