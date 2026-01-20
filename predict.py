import tensorflow as tf
import cv2
import numpy as np

# Load trained model
model = tf.keras.models.load_model("image_model.h5")

# CIFAR-10 class names
class_names = ['airplane','car','bird','cat','deer','dog','frog','horse','ship','truck']

# Load your test image (replace 'test.jpg' with your image file)
img = cv2.imread("test.jpg")
img = cv2.resize(img, (32,32))  # Must match CIFAR-10 input size
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Make prediction
pred = model.predict(img)
result = class_names[np.argmax(pred)]

print("Prediction:", result)
