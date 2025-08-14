import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Flatten

#Avoid out of memory errors by setting gpu memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)
    
print("Num GPUs avaliable: ", len(gpus))

#setting up the paths for the folder structure
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')