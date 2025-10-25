import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import joblib
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
cnn_model = load_model("model/cnn_model.h5")
svm_model = joblib.load("model/svm_model.joblib")

# Feature extractor (Dense(128))
feature_extractor = tf.keras.Model(inputs=cnn_model.input,
                                   outputs=cnn_model.layers[-3].output)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file:
            return render_template("index.html", label=None, confidence=None, error="Please upload an image!")

        # Save uploaded image
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        # Preprocess image for CNN
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = np.expand_dims(img, axis=0) / 255.0

        # CNN prediction
        cnn_pred = float(cnn_model.predict(img)[0][0])

        # SVM prediction (CNN features)
        svm_features = feature_extractor.predict(img)
        svm_pred = float(svm_model.predict_proba(svm_features)[0][1])

        # Average confidence
        avg_confidence = (cnn_pred + svm_pred) / 2
        label = "Fake" if avg_confidence > 0.5 else "Real"

        return render_template(
            "index.html",
            label=label,
            confidence=round(avg_confidence * 100, 2),
            image_path=img_path.replace("\\", "/")
        )

    return render_template("index.html", label=None, confidence=None)

if __name__ == "__main__":
    app.run(debug=True)
