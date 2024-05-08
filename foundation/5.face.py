import cv2

faceCascade= cv2.CascadeClassifier("haarcascades\haarcascade_frontalface_default.xml")      #reading the trained model of face

img = cv2.imread('demo\oin.jpg')
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)              #converting the image to grayscale

faces = faceCascade.detectMultiScale(imgGray,2,2)           #detecting the co-ordinates of strt and end position of

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)


cv2.imshow("Result", img)
cv2.waitKey(0)