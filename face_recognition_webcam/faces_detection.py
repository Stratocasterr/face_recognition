from asyncio import events
import cv2 as cv
import numpy as np
import os
import json
from config import *
from face_trainer import *

project_files = os.listdir(PROJECT_DICT_PATH)                                     
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
    u_confidence = UP_CONFIDENCE
    d_confidence = DOWN_CONFIDENCE
    changing_upper_confidence = False
    changing_down_confidence = False
    show_info = False
    
    while running:
        ret, color_frame = video.read()
        print(show_info)
        gray_frame = cv.cvtColor(color_frame, cv.COLOR_BGR2GRAY)                                                          # convert video frame to grayscale
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        
        

        for (x, y, w, h) in faces:
            if DISPLAY_FRAME: cv.rectangle(color_frame, (x, y), (x + w, y + h), FRAME_COLOR, FRAME_WIDTH)
            roi_gray = gray_frame[y:y+h, x:x+w]
          
            
            id, confidence = face_recognizer.predict(roi_gray)                                                             # confidence is a distance between compared images' histograms
            if confidence >= d_confidence and confidence <= u_confidence:
                predicted_name = names[id]

                cv.putText(color_frame, predicted_name, 
                    (x, y), 
                    cv.FONT_HERSHEY_COMPLEX_SMALL, 
                    FONT_SIZE, 
                    FONT_COLOR, 
                    3, 
                    cv.LINE_AA)
    
       
        if cv.waitKey(20) == ord('q'):
            running = False
            return 0
        
        elif cv.waitKey(1) == ord('s'):
            if not show_info:
                show_info = True
            else: show_info = False
        

        elif cv.waitKey(1) == ord('u'): 
            changing_upper_confidence = True
            changing_down_confidence = False
        elif cv.waitKey(1) == ord('d'): 
            changing_upper_confidence = False
            changing_down_confidence = True
    
        elif cv.waitKey(1) == ord('='): 
            if changing_down_confidence: d_confidence += 1
            elif changing_upper_confidence: u_confidence += 1
            
        elif cv.waitKey(1) == ord('-'):
            if changing_down_confidence: d_confidence -= 1
            elif changing_upper_confidence: u_confidence -= 1

        if show_info:
            cv.rectangle(color_frame, INFO_RECTANGLE_POSITION, 
            (INFO_RECTANGLE_POSITION[0] + INFO_RECTANGLE_SIZE[0], INFO_RECTANGLE_POSITION[1] + INFO_RECTANGLE_SIZE[1]),
            INFO_RECTANGLE_COLOR, -1)

            if changing_down_confidence:
                cv.putText(color_frame, "Changing down confidence...", 
                    (10, 50), 
                    cv.FONT_HERSHEY_COMPLEX_SMALL, 
                    FONT_SIZE*1.3, 
                    FONT_COLOR, 
                    3, 
                    cv.LINE_AA)

            elif changing_upper_confidence:
                cv.putText(color_frame, "Changing up confidence...", 
                        (10, 50), 
                        cv.FONT_HERSHEY_COMPLEX_SMALL, 
                        FONT_SIZE*1.3, 
                        FONT_COLOR, 
                        3, 
                        cv.LINE_AA)

            cv.putText(color_frame, "Down confidence: " + str(d_confidence), 
                        (10, 100), 
                        cv.FONT_HERSHEY_COMPLEX_SMALL, 
                        FONT_SIZE *1.3, 
                        FONT_COLOR, 
                        3, 
                        cv.LINE_AA)

            cv.putText(color_frame, "Up confidence: " + str(u_confidence), 
                        (10, 150), 
                        cv.FONT_HERSHEY_COMPLEX_SMALL, 
                        FONT_SIZE*1.3, 
                        FONT_COLOR, 
                        3, 
                        cv.LINE_AA)

        cv.imshow("Face Recognition",color_frame)


    video.release()
    cv.destroyAllWindows()

