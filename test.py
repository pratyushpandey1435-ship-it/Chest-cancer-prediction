import tensorflow as tf

# Check if TensorFlow can access a GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Print available GPU details
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU Details:")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU found. TensorFlow is using CPU.")

print("Is TensorFlow using GPU?", tf.test.is_built_with_cuda())
print("GPU Available:", tf.config.experimental.list_physical_devices('GPU'))
print("TensorFlow Version:", tf.__version__)
