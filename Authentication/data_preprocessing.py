import tensorflow as tf
import os 

#Loading images into a data pipeline and scale/resize images 

POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

#grab images from respective directories
anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(300)

#file path taken from directories above
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path) #read image
    img = tf.io.decode_jpeg(byte_img) #load image
    img = tf.image.resize(img, (100,100)) #resize image (100x100x3)
    img = img/255.0 #scales image between 0 and 1 pixel value is from 0 to 255

    return img


#creating the labelled dataset - positive for 1 and negative for 0 (tuples or twins)
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
data = positives.concatenate(negatives) #joining positives and negatives together

#The build and train partition
def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)


'''data holds three values an anchor, a positive or a negative and a label being either 0 or 1'''
#dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024) #shuffles images for training

#training partition
train_data = data.take(round(len(data)*.7)) #70% as the training partition for the images
train_data = train_data.batch(16) #passing data as batches of 16
train_data = train_data.prefetch(8) #starts preprocessing the next batch of data to avoid bottlenecks

#testing partition
test_data = data.skip(round(len(data)*.7)) #skip the training data
test_data = test_data.take(round(len(data)*.3)) #grabs the remaining 30% for testing
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)