import tensorflow as tf
from data_preprocessing import train_data

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {[gpu.name for gpu in gpus]}")
else:
    print("No GPU detected. Using CPU instead.")

# Optional: check TensorFlow version
print("TensorFlow version:", tf.__version__)


# Quick check of a batch
test_input, test_val, y = next(train_data.as_numpy_iterator())
print("Sample labels in first batch:", y[:20])
