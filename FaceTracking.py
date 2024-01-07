import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

FaceDetection = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalface_default.xml')

def DetectFace(image): 
    FaceImage = image.copy()

    RectOfFace = FaceDetection.detectMultiScale(FaceImage)
    
    for (x,y,width,height) in RectOfFace:
        cv2.rectangle(FaceImage,(x,y),(x+width,y+height),(255,255,255),10)

    return FaceImage


Capture = cv2.VideoCapture(0)

while True: 
    ret,frame = Capture.read(0)
    frame = DetectFace(frame)
    cv2.imshow('Face Detection',frame)

    Key = cv2.waitKey(1)
    if Key == 27: 
        break

Capture.release()
cv2.destroyAllWindows()

