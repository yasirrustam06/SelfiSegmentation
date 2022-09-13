import cv2
import mediapipe as mp
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os



cap=cv2.VideoCapture(0)

cap.set(3,480)
cap.set(4,684)

segmentor=SelfiSegmentation()

fpsReader=cvzone.FPS()

while True:
    ret,img=cap.read()
    imgout=segmentor.removeBG(img,(255,0,0),threshold=0.8)
    imgStacked = cvzone.stackImages([img,imgout],1,2)
    _,imgStacked=fpsReader.update(imgStacked,color=(0,255,255))


    cv2.imshow("imgStacked",imgStacked)

    cv2.waitKey(1)

