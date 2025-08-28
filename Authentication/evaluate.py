#file for evaluating the model after training
from data_preprocessing import test_data #test data partition
from tensorflow.keras.metrics import Precision, Recall
from model import make_siamese_model
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

siamese_model = make_siamese_model()

#restoring checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint = tf.train.Checkpoint(siamese_model=siamese_model)
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint).expect_partial()
    print(f"Restored model from {latest_checkpoint}")
else:
    print("!!!!No checkpoint was found, evaluating untrained model!!!!")

# gets a batch of testing data input, validation data, labels
test_input, test_val, y_true = test_data.as_numpy_iterator().next()

#making predictions for this batch
y_pred = siamese_model.predict([test_input, test_val])

# DEBUG: print raw predictions vs labels
print("\n=== Sample predictions vs labels (first 20) ===")
for i in range(min(20, len(y_pred))):
    print(f"Prediction: {y_pred[i][0]:.4f} | Thresholded: {1 if y_pred[i][0] > 0.5 else 0} | True: {int(y_true[i])}")

#adding a threshold 
y_pred_binary = (y_pred > 0.5).astype(int)

# visualize results for each item in the batch
for i in range(len(test_input)):
    plt.figure(figsize=(6,3))

    # input image
    plt.subplot(1,2,1)
    plt.imshow(test_input[i])
    plt.title("Input")
    plt.axis("off")

    # validation image
    plt.subplot(1,2,2)
    plt.imshow(test_val[i])
    plt.title("Validation")
    plt.axis("off")

    # print label + prediction in console
    print(f"Pair {i+1}: True Label = {y_true[i]}, Prediction = {y_pred_binary[i][0]}")

    # show figure
    plt.suptitle(f"True: {y_true[i]} | Pred: {y_pred_binary[i][0]}")
    plt.show()

# creating metrics objects
r = Recall()
p = Precision()

# looping through testing dataset
for test_input, test_val, y_true in test_data.as_numpy_iterator():
    y_pred = siamese_model.predict([test_input, test_val])
    y_binary = (y_pred > 0.5).astype(int)
    r.update_state(y_true, y_binary)
    p.update_state(y_true, y_binary)

print("Recall:", r.result().numpy())
print("Precision:", p.result().numpy())