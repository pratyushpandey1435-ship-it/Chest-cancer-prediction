import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.model = load_model(os.path.join("artifacts", "training", "model.h5"))

    def predict(self, image_path=None):
        # Use provided image path if given
        imagename = image_path if image_path else self.filename

        # Load and preprocess the image
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Normalize

        # Predict
        predictions = self.model.predict(test_image)
        print("Raw model output:", predictions)

        # Determine result
        result = np.argmax(predictions, axis=1)[0] if predictions.shape[1] > 1 else int(predictions[0][0] > 0.5)

        # Class Labels
        class_labels = [
            "Adenocarcinoma (Left Lower Lobe)",
            "Large Cell Carcinoma (Left Hilum)",
            "Normal",
            "Squamous Cell Carcinoma (Left Hilum)"
        ]

        return class_labels[result]
