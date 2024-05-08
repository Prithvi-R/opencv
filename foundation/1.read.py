import cv2

#-------------------------
frameWidth = 640
frameHeight = 480
#--------------------------

######################## CREATE IMAGE ############################
import numpy as np
img = np.zeros((512,512,3),np.uint8)    #create an image with custom colors


######################## READ IMAGE ############################

img = cv2.imread("Resources/lena.png")      #Read images
img = cv2.resize(img, (frameWidth, frameHeight))           #resize as per choice
cv2.imshow("Lena Soderberg",img)            #Display it
cv2.waitKey(0)             #it is basically for how long a frame should be displayes (time in milliseconds) --> enter 0 for forever


######################### READ VIDEO #############################

vid = cv2.VideoCapture("Resources/test_ video.mp4")     #Read videos

while True:             # to display videos we have to display them frame-by-frame

    success, img = vid.read()                                  #read current frame
    img = cv2.resize(img, (frameWidth, frameHeight))           #resize
    cv2.imshow("Result", img)                                  #display

    if cv2.waitKey(1) & 0xFF == ord('q'):         #display the frame for 1-millisecond and if the keyboard stroke is = q (stop displaying the video)
        break


######################### READ WEBCAM  ############################
cap = cv2.VideoCapture(0)       #here {numerc_value} stand for camera   0-primary webcam   1-additional webcam 

cap.set(3, frameWidth)      #its function to set width size
cap.set(4, frameHeight)     #function to set height size
cap.set(10, 150)
while True:
    success, img = cap.read()
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break