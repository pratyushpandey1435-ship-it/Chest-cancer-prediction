import os

import numpy as np
import pydicom
import tensorflow as tf


def preprocess_dcm(file_path):
    dicom = pydicom.dcmread(file_path)
    image = dicom.pixel_array.astype(float)
    if image.max() != 0:
        image = image / image.max()
    if len(image.shape) == 2:
        image = image[..., np.newaxis]
    if len(image.shape) != 3:
        raise ValueError(f"Unexpected image shape: {image.shape}")
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32)
    return image.numpy()


def load_data(img_dir, categories):
    data = []
    labels = []
    total_files = 0
    processed_files = 0

    for label, category in enumerate(categories):
        category_path = os.path.join(img_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: Category path does not exist: {category_path}")
            continue

        files = [f for f in os.listdir(category_path) if f.endswith(".dcm")]
        total_files += len(files)

        for file_name in files:
            file_path = os.path.join(category_path, file_name)
            try:
                image = preprocess_dcm(file_path)
                data.append(image)
                labels.append(label)
                processed_files += 1
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

    print(f"Successfully processed {processed_files} out of {total_files} files")

    if not data:
        raise ValueError("No images were successfully processed")

    return np.array(data), np.array(labels)


def create_tf_dataset(X, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(len(X)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
