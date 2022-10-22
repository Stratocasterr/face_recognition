import cv2 as cv
import face_recognition as fr
import numpy as np
import os
import pickle

training_images = r'C:/Users/kacpe/Visual Studio Projects/face_recognition/train_friends_images/'


face_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_frontalface_default.xml')
face_recognizer = cv.face.LBPHFaceRecognizer_create()

current_id=0
names_ids = {}
encodes_of_known = []
ids = []
names_of_known = []
faces_to_train = []
for dirpath, dirnames, filenames in os.walk(training_images):
       
        for dirname in dirnames:
            for dp, dirs, files in os.walk(training_images+ str(dirname)):
                print(dirname)
                for index, file in enumerate(files):
                    name = dp.split("/")[-1]
                    names_of_known.append(name)
                    file_path = os.path.join(dp, file)
                    #print(file_path, name)
                    if name not in names_ids:
                        names_ids[name] = current_id
                        current_id +=1
                    
                    id = names_ids[name]
 
                    gray_image = cv.imread(file_path,cv.IMREAD_GRAYSCALE)
                    #gray_image = cv.cvtColor(image, cv.COLOR_BGR2)
                    gray_image_array = np.array(gray_image, "uint8")

                    faces = face_cascade.detectMultiScale(gray_image_array, 1.12, 5)
                    
                    for (x, y, w, h) in faces:
                        roi_gray_image = gray_image[y:y+h, x:x+w]
                        faces_to_train.append(roi_gray_image)
                        ids.append(id)
                        cv.rectangle(gray_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.imshow("Ds", gray_image)
                    #cv.waitKey(0)
                    cv.destroyAllWindows()
                    #cv.waitKey(0)
                    #print(names_ids)

with open("names.pickle", 'wb') as f:
    pickle.dump(names_ids, f)

face_recognizer.train(faces_to_train, np.array(ids))
face_recognizer.save("trainner.yml")