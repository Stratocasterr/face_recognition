import cv2 as cv
import numpy as np
import os
import json
from config import *
from face_trainer import *




project_files = os.listdir(PROJECT_DICT_PATH)


                                                  # start training process


if any([TRAINING_DATA_FILE_NAME not in project_files, NAMES_OF_PEOPLE_FILE_NAME not in project_files]) : face_recognize_trainer()

face_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_frontalface_default.xml')
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("trainner.yml")



names = {}
with open("names_with_ids.json", 'r') as file:                                                                             # transfer names and ids 
    names = json.load(file)
    names = {v:k for k,v in names.items()}
    file.close()



video = cv.VideoCapture(0)                                                                                                 # capture video cam frames and set parameters
video.set(cv.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)                                                                                   
video.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)


def recognize_people_on_webcam(run):
    running = run
    while running:
        ret, color_frame = video.read()
        
        gray_frame = cv.cvtColor(color_frame, cv.COLOR_BGR2GRAY)                                                          # convert video frame to grayscale
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        
        for (x, y, w, h) in faces:
            if DISPLAY_FRAME: cv.rectangle(color_frame, (x, y), (x + w, y + h), FRAME_COLOR, FRAME_WIDTH)
            roi_gray = gray_frame[y:y+h, x:x+w]
          
            id, confidence = face_recognizer.predict(roi_gray)
            if confidence >= CONFIDENCE :
                predicted_name = names[id]

                cv.putText(color_frame, predicted_name, 
                    (x, y), 
                    cv.FONT_HERSHEY_COMPLEX_SMALL, 
                    FONT_SIZE, 
                    FONT_COLOR, 
                    3, 
                    cv.LINE_AA)
       

        cv.imshow("Face Recognition",color_frame)
        if cv.waitKey(20) & 0xFF == ord('q'):
            running = False
            return 0


    video.release()
    cv.destroyAllWindows()

recognize_people_on_webcam(RUN_CAMERA)