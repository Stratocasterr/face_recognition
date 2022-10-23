import cv2 as cv
import face_recognition as fr
import numpy as np
import os
import pickle as pkl


face_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_frontalface_default.xml')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("trainner.yml")

names = {}
with open("names.pickle", 'rb') as f:
    names = pkl.load(f)
    #names = {v:k for k,v in names.items()}
print(names)
video = cv.VideoCapture(0)

# learning


    
# find people
def recognize_people_on_webcam():
    
    run = False
    while run:
        ret, color_frame = video.read()
        
        gray_frame = cv.cvtColor(color_frame, cv.COLOR_BGR2GRAY)                                                          # convert video frame to grayscale
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv.rectangle(color_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray_frame[y:y+h, x:x+w]
          
            id, confidence = face_recognizer.predict(roi_gray)
            if confidence >=45 and confidence <=85:
                predicted_name = names[id]

                cv.putText(color_frame, predicted_name, 
                    (x, y), 
                    cv.FONT_HERSHEY_COMPLEX_SMALL, 
                    0.8, 
                    (0, 255, 0), 
                    3, 
                    cv.LINE_AA)
       

        cv.imshow("Face Recognition",color_frame)
        if cv.waitKey(20) & 0xFF == ord('q'):
            run = False


    video.release()
    cv.destroyAllWindows()

recognize_people_on_webcam()