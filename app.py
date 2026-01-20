import os
import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
from flask import Flask, render_template, request

# Hide TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

model = tf.keras.models.load_model("image_model.h5")

class_names = ['airplane','car','bird','cat','deer','dog','frog','horse','ship','truck']

@app.route('/', methods=['GET','POST'])
def index():
    prediction = None
    image_file = None
    chart_file = None

    if request.method == 'POST':
        file = request.files['image']

        if not os.path.exists("static"):
            os.makedirs("static")

        filepath = os.path.join("static", file.filename)
        file.save(filepath)
        image_file = file.filename

        img = cv2.imread(filepath)
        img = cv2.resize(img, (32,32))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        idx = np.argmax(pred)
        prediction = class_names[idx]
        confidence = np.max(pred) * 100

        # ---- Text To Speech (female voice safely) ----
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for voice in voices:
            if "Zira" in voice.name:
                engine.setProperty('voice', voice.id)
                break
        engine.setProperty('rate', 150)
        engine.say(f" It is {prediction}")
        engine.runAndWait()
        del engine
        # --------------------------------------------

        # Bar chart
        plt.figure(figsize=(8,4))
        plt.bar(class_names, pred[0])
        plt.xticks(rotation=45)
        plt.title("Prediction Confidence")
        plt.tight_layout()

        chart_name = "chart.png"
        chart_path = os.path.join("static", chart_name)
        plt.savefig(chart_path)
        plt.close()

        chart_file = chart_name

    return render_template("index.html",
                           prediction=prediction,
                           image_file=image_file,
                           chart_file=chart_file)

if __name__ == '__main__':
    app.run(debug=True)
