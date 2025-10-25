import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tqdm import tqdm

# Paths and config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMG_SIZE = (128, 128)
BATCH_SIZE = 16

# Data generators
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)
val_generator = datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=True
)
test_generator = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
    os.path.join(DATASET_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# CNN model
inputs = Input(shape=(*IMG_SIZE, 3))
x = Conv2D(32, (3,3), activation='relu')(inputs)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

cnn_model = Model(inputs=inputs, outputs=outputs)
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(train_generator, validation_data=val_generator, epochs=5)

# Save CNN model
os.makedirs("model", exist_ok=True)
cnn_model.save("model/cnn_model.h5")

# Feature extractor for SVM (Dense(128) output)
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-3].output)

def extract_features(generator):
    features, labels = [], []
    for _ in tqdm(range(len(generator)), desc="Extracting features"):
        x_batch, y_batch = next(generator)
        feat = feature_extractor.predict(x_batch)
        features.append(feat)
        labels.append(y_batch)
    return np.vstack(features), np.hstack(labels)

X_train, y_train = extract_features(train_generator)
X_test, y_test = extract_features(test_generator)

# Train SVM on CNN features
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Evaluate
y_pred = svm_model.predict(X_test)
print("\n📊 SVM Classification Report:")
print(classification_report(y_test, y_pred))
print(f"✅ SVM Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save SVM model
joblib.dump(svm_model, "model/svm_model.joblib")
