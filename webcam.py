import numpy as np
import cv2 as cv
import os
import face_recognition as fr
base_image_path = 'C:/Users/kacpe/Visual Studio Projects/face_recognition/friends_images/'
train_image_path = base_image_path + 'Kacper/kacper0.jpg'

def huj():

    video = cv.VideoCapture(0)
    run = True
    image = cv.imread(train_image_path)
    print(image)
    print("vweeqw")
       
    while run:
        ret, color_frame = video.read()
        print(color_frame)
        faces = []
        gray_frame = cv.cvtColor(color_frame, cv.COLOR_BGR2GRAY)
        
 
        face_locations = fr.face_locations(gray_frame)
        face_encode = fr.face_encodings(gray_frame, face_locations)

        
