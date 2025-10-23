import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('emotion_detector_model.h5')

# Load a face image
face = cv2.imread('test_face.jpg', cv2.IMREAD_GRAYSCALE)
face = cv2.resize(face, (48, 48))
face = face / 255.0
face = np.expand_dims(face, axis=0)
face = np.expand_dims(face, axis=-1)

# Predict emotion
pred = model.predict(face)
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
print("Predicted Emotion:", emotion_classes[np.argmax(pred)])
