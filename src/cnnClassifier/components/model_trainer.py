import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None 

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Reset optimizer
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Validation data generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)

        # self.valid_generator = valid_datagenerator.flow_from_directory(
        #     directory=self.config.validation_data,
        #     target_size=self.config.params_image_size[:-1],
        #     batch_size=self.config.params_batch_size,
        #     class_mode="categorical",  # Fixed: Ensures multi-class classification
        #     shuffle=False
        # )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.validation_data,
            **dataflow_kwargs,  # UPDATED: Used **dataflow_kwargs to simplify
            class_mode="categorical",  
            shuffle=False
        )

        # Training data generator
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        # self.train_generator = train_datagenerator.flow_from_directory(
        #     directory=self.config.training_data,
        #     target_size=self.config.params_image_size[:-1],
        #     batch_size=self.config.params_batch_size,
        #     class_mode="categorical",  # Fixed: Ensures correct class label handling
        #     shuffle=True
        # )

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            **dataflow_kwargs,  # UPDATED: Used **dataflow_kwargs
            class_mode="categorical",
            shuffle=True
        )
        print("Train Class Indices:", self.train_generator.class_indices)
        print("Train Num Classes:", self.train_generator.num_classes)
        print("Valid Class Indices:", self.valid_generator.class_indices)
        print("Valid Num Classes:", self.valid_generator.num_classes)

    
    def save_model(self,path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        if not hasattr(self.model, "optimizer"):
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Reset optimizer
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
        
