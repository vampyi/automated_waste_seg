# Dependencies
import numpy as np
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16
import tensorflow as tf

# Function to predict the class of the image
def getPrediction(filename):
    # Load trained model
    model = tf.keras.models.load_model("/Users/vansh/Desktop/SORT/Resources/Model/final_model_weights.hdf5")
    
    # Load and preprocess the image
    img_path = f"/Users/vansh/Desktop/SORT/static/{filename}"
    img = load_img(img_path, target_size=(180, 180))  # Resize image
    img = img_to_array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img)  # Get class probabilities
    category = np.argmax(predictions, axis=1)[0]  # Get predicted class index

    # Assign class labels
    if category == 1:
        answer = "Recycle"
        probability_results = float(predictions[0][1])  # Probability of Recycle class
    else:
        answer = "Organic"
        probability_results = float(predictions[0][0])  # Probability of Organic class

    return str(answer), str(probability_results), filename

