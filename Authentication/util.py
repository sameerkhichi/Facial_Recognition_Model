#this file is only needed for building the exe
import tensorflow as tf


def preprocess(file_path):
    byte_img = tf.io.read_file(file_path) #read image
    img = tf.io.decode_jpeg(byte_img) #load image
    img = tf.image.resize(img, (100,100)) #resize image (100x100x3)
    img = img/255.0 #scales image between 0 and 1 pixel value is from 0 to 255

    return img