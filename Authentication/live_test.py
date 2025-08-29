import os
import tensorflow as tf
from data_preprocessing import preprocess
import numpy as np
import cv2
from model import L1Dist

#load trained model from h5 file
siamese_model = tf.keras.models.load_model(
    "siamesemodelv2.h5",
    custom_objects={"L1Dist": L1Dist, "BinaryCrossentropy": tf.losses.BinaryCrossentropy}
)

siamese_model.summary()  #confirm it loaded

def verify(model, detection_threshold, verification_threshold):

    #results array
    results = []
    for image in os.listdir(os.path.join('live_test_data', 'verification_images')):
        input_img = preprocess(os.path.join('live_test_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('live_test_data', 'verification_images', image))

        #make predictions
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    #detection threshold: a metric to determine if a prediction is considered positive
    detection = np.sum(np.array(results) > detection_threshold)

    #verification threshold - proportion of positive predictions / total positive samples
    verification = detection / len(os.listdir(os.path.join('live_test_data', 'verification_images')))
    verified = verification > verification_threshold

    print(f"Verification: {verified} ({verification*100:.2f}% match)")

    return results, verified, verification


#get live webcam feed for verificaiton
cap = cv2.VideoCapture(0)
overlay_text = "Press 'v' to verify"

while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120+250,200:200+250, :] #cut down frame to 250x250 pixels
    
    #overlay text on video
    cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Verification', frame)

    #verification trigger key - press v
    if cv2.waitKey(10) & 0xFF == ord('v'): 

        #save input image from webcam to folder
        cv2.imwrite(os.path.join('live_test_data', 'input_image', 'input_image.jpg'), frame)

        #run verification
        results, verified, verification = verify(siamese_model, 0.7, 0.5) #both thresholds of 0.5 - passing in the model
        
        #update overlay text with result
        if verified:
            overlay_text = f"Verified ({verification*100:.2f}%)"
        else:
            overlay_text = f"Not Verified ({verification*100:.2f}%)"

    #q to quit and close
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()