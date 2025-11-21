import tensorflow as tf
import mlflow
from pathlib import Path
from urllib.parse import urlparse
from dataclasses import dataclass
import json
import os
import pickle
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json, read_yaml, create_directories

# Set MLflow tracking URI to DagsHub
mlflow.set_tracking_uri("https://dagshub.com/AryanDhanuka10/End-To-End-ML_Project-Chest-Cancer-Detection-Using-MLOps-and-DVC.mlflow")

# Ensure authentication is set for DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = "AryanDhanuka10"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "e83049708341d244ec2c9f994ec79046028731de"



@dataclass
class EvaluationConfig:
    path_of_model: Path
    training_data: str
    params_image_size: tuple
    params_batch_size: int
    mlflow_url: str
    all_params: dict

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.30  # Keep validation split
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="categorical"  # Ensure proper multi-class classification
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        print("Valid Class Indices:", self.valid_generator.class_indices)
        print("Detected classes:", self.valid_generator.class_indices)
        print("Number of detected classes:", len(self.valid_generator.class_indices))
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        model = tf.keras.models.load_model(path)
        model.summary()  # Ensure model supports 4 classes
        return model

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()

        # Ensure 4 output classes
        num_classes = len(self.valid_generator.class_indices)
        if num_classes != 4:
            raise ValueError(f"Expected 4 classes, but found {num_classes}. Check dataset!")

        self.score = self.model.evaluate(self.valid_generator)
        print(f"Loss: {self.score[0]}, Accuracy: {self.score[1]}")
        self.save_score()

    import json

    def save_json(path: Path, data: dict):
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_url)
        mlflow.set_tracking_uri("https://dagshub.com/AryanDhanuka10/End-To-End-ML_Project-Chest-Cancer-Detection-Using-MLOps-and-DVC.mlflow")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model")
            else:
                mlflow.keras.log_model(self.model, "model")
    

