# Import Dependencies
import json

import joblib
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow.keras import layers, models

from utils import create_tf_dataset, load_data
from constants import *

# # Tabular Data Loading and Splitting
# print("Loading tabular data...")
# csv_data = pd.read_csv(CSV_PATH)
# X_csv = csv_data.drop("DEATH_EVENT", axis=1)
# y_csv = csv_data["DEATH_EVENT"]

# X_train_csv, X_test_csv, y_train_csv, y_test_csv = train_test_split(
#     X_csv, y_csv, test_size=0.2, random_state=42
# )
# print("Tabular data loaded and split successfully.")

# # RandomForest Model Training and Hyperparameter Tuning
# print("Starting RandomForest model training with hyperparameter tuning...")
# rf_params = {
#     "n_estimators": [50, 100, 200, 500, 1000, 5000],
#     "max_depth": [None, 10, 20, 50, 100, 1000],
#     "min_samples_split": [2, 5, 10, 20, 50, 500],
# }
# rf_model = RandomForestClassifier(random_state=42)
# grid_search = GridSearchCV(rf_model, rf_params, cv=3, scoring="accuracy", n_jobs=-1)
# grid_search.fit(X_train_csv, y_train_csv)
# print("RandomForest model training completed.")

# # Best model
# best_rf_model = grid_search.best_estimator_
# print(f"Best RandomForest parameters: {grid_search.best_params_}")

# # RandomForest Model Evaluation
# print("Evaluating RandomForest model...")
# y_pred_csv = best_rf_model.predict(X_test_csv)
# rf_scores = {
#     "accuracy": accuracy_score(y_test_csv, y_pred_csv),
#     "f1_score": f1_score(y_test_csv, y_pred_csv),
#     "classification_report": classification_report(
#         y_test_csv, y_pred_csv, output_dict=True
#     ),
# }
# print(f"RandomForest Evaluation Scores: {rf_scores}")

# # Save RF model and scores
# print("Saving RandomForest model and evaluation scores...")
# with open(RF_SCORE, "w") as rf_score_file:
#     json.dump(rf_scores, rf_score_file)
# joblib.dump(best_rf_model, RF_MODEL)
# print("RandomForest model and scores saved successfully.")

# Image Data Loading and Preprocessing
print("Loading and preprocessing image data...")
data, labels = load_data(IMG_DIR, CATEGORIES)
print(data)
print(data.shape)
X_train_img, X_temp, y_train_img, y_temp = train_test_split(
    data, labels, test_size=0.3, random_state=42
)
X_val_img, X_test_img, y_val_img, y_test_img = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
print("Image data loaded and split successfully.")

train_img = create_tf_dataset(X_train_img, y_train_img)
val_img = create_tf_dataset(X_val_img, y_val_img)
test_img = create_tf_dataset(X_test_img, y_test_img)
print("TensorFlow datasets created for training, validation, and testing.")

# TensorFlow Model Training
print("Starting CNN model training...")
cnn_model = models.Sequential(
    [
        layers.InputLayer(input_shape=(224, 224, 1)),
        layers.Conv2D(8, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D( 16, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dense(len(CATEGORIES), activation="softmax"),
    ]
)

cnn_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Use ModelCheckpoint to save the best model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    CNN_MODEL, monitor="val_accuracy", save_best_only=True, mode="max"
)

cnn_model.fit(train_img, validation_data=val_img, epochs=10, callbacks=[checkpoint])
print("CNN model training completed and best model saved.")

# TensorFlow Model Evaluation
print("Evaluating CNN model...")
cnn_model = tf.keras.models.load_model(CNN_MODEL)  # Load the best model
cnn_eval = cnn_model.evaluate(test_img)
cnn_scores = {
    "loss": cnn_eval[0],
    "accuracy": cnn_eval[1],
}
print(f"CNN Evaluation Scores: {cnn_scores}")

# Save CNN scores
print("Saving CNN evaluation scores...")
with open(CNN_SCORE, "w") as cnn_score_file:
    json.dump(cnn_scores, cnn_score_file)
print("CNN scores saved successfully.")
