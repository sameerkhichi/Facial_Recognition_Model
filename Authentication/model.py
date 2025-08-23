#Researched paper used: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

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

#siamese L1 distance layer - tells us how similar our images are - using anchor and positive/negative
class L1Dist(Layer):

    #init method for inheritance - kwargs makes exporting easier
    def __init__(self, **kwargs):
        super().__init__()

    #similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding) #returning the difference

#embedding layer - gives us a model - 4096 output layer
def make_embedding():
    inp = Input(shape=(100,100,3), name='input_image') #specifying the shape of the input image
    
    #note filters scan and detect features in the image and pass to deeper layers.

    #First block layer 
    c1 = Conv2D(64, (10,10), activation='relu')(inp) #64 filters 10x10 pixels relu activation - passing input image to convolutional layer
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1) #max pooling layer - 64 units 2x2 area

    #second layer - values for filters and shape taken from research paper
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)

    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)

    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4) #flattening the elements - single dimension rather than 6x6x256
    d1 = Dense(4096, activation='sigmoid')(f1) #feature vector with sigmoid activation

    return Model(inputs=[inp], outputs=[d1], name='embedding')

def make_siamese_model():

    #anchor image input in the network
    input_image = Input(name='input_img', shape=(100, 100, 3))

    #validation image input in the network
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    #shared embedding network
    embedding = make_embedding()

    #combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'

    #encoding both input images - the distances
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    #combines distances - figures if the distances are close enough to call the images the same - using sigmoid activation
    #4096 units in and one unit out 1/0
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


#for debugging and prototyping
if __name__ == "__main__":
    embedding = make_embedding()
    embedding.summary()

    siamese_model = make_siamese_model()
    siamese_model.summary()