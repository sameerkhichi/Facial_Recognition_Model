#generates unique image names
import uuid
import os
import cv2

POS_PATH = os.path.join('data', 'positive')
ANC_PATH = os.path.join('data', 'anchor')

#creating the unique name for the image to store into the anchor folder
os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))

#opening a connection to the webcam
cap = cv2.VideoCapture(4)
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
    
    #Collect positives
    if cv2.waitKey(1) & 0XFF == ord('p'):
        #Create the unique file path 
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        #Write out positive image
        cv2.imwrite(imgname, frame)
    
    #reflect image to screen
    cv2.imshow('Image Collection', frame)
    
    #allowing quit
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
        
#releasing webcam to avoid lag
cap.release()
cv2.destroyAllWindows()