#this files purpose is to collect data from the webcam for training purposes - not a part of the app

#generates unique image names
import uuid
import os
import cv2
import tensorflow as tf

POS_PATH = os.path.join('data', 'positive')
ANC_PATH = os.path.join('data', 'anchor')

#this will fill up quickly with larger sets of data so dont exceed it by too much
new_images = []

#creating the unique name for the image to store into the anchor folder
os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))

#opening a connection to the webcam
cap = cv2.VideoCapture(0)
while cap.isOpened(): 
    ret, frame = cap.read()
   
    #Cut down frame to 250x250px
    frame = frame[120:120+250,200:200+250, :]
    
    #Collect anchors 
    if cv2.waitKey(1) & 0XFF == ord('a'):
        #Create the unique file path 
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        #Write out anchor image
        cv2.imwrite(imgname, frame)
        new_images.append(imgname)
    
    #Collect positives
    if cv2.waitKey(1) & 0XFF == ord('p'):
        #Create the unique file path 
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        #Write out positive image
        cv2.imwrite(imgname, frame)
        new_images.append(imgname)
    
    #reflect image to screen
    cv2.imshow('Image Collection', frame)
    
    #allowing quit
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
        
#releasing webcam to avoid lag
cap.release()
cv2.destroyAllWindows()


#TensorFlow augmentation pipeline with extra brightness/saturation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomBrightness(factor=0.2),   #±20% brightness
    tf.keras.layers.RandomSaturation(factor=0.2)    #±20% saturation
])

def augment_image_data(image_paths):
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Expand dims to make it batch of 1
        img_tensor = tf.expand_dims(img_rgb, axis=0)
        aug_img_tensor = data_augmentation(img_tensor)
        aug_img = tf.cast(aug_img_tensor[0], tf.uint8).numpy()

        #Save augmented image with random JPEG quality
        folder = os.path.dirname(img_path)
        aug_filename = os.path.join(folder, f"aug_{uuid.uuid1()}.jpg")
        jpeg_quality = int(tf.random.uniform([], 60, 100).numpy())  #Random between 60-100
        cv2.imwrite(aug_filename, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR), 
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

#Augment only new images from current session to avoid large file
if new_images:
    print(f"Running data augmentation on {len(new_images)} images...")
    augment_image_data(new_images)
    print("Data augmentation complete.")
else:
    print("No images captured this session. Skipping augmentation.")