import cv2 as cv
import face_recognition as fr
import numpy as np
import os
import pickle as pkl

training_images = r'C:/Users/kacpe/Visual Studio Projects/face_recognition/train_friends_images/'


face_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_frontalface_default.xml')
face_recognizer = cv.face.LBPHFaceRecognizer_create()

current_id=0
names_ids = {}

ids = []
names_of_known = []
faces_to_train = []


def change_names(path, start_index, files, dirname):
    index = start_index + 1
    small_name = dirname.lower()
    for f in files:
        if small_name not in f:
            os.rename(path  + "/" + f, path + "/"+ small_name + str(index) + ".jpg")
            index += 1


for dirpath, dirnames, filenames in os.walk(training_images):
        for dirname in dirnames:
            for dp, dirs, files in os.walk(training_images+ str(dirname)):

                print(dirname)
                name_biggest_index = 0                                                                      # for changing files' names

                for index, file in enumerate(files):
                    name = dp.split("/")[-1]
                    names_of_known.append(name)
                    file_path = os.path.join(dp, file)

                    if dirname.lower() in file:                                                             # for changing files' names
                        number = int(file.split(".")[0].replace(dirname.lower(), ""))
                        if number > name_biggest_index: name_biggest_index = number

                   
                    if name not in names_ids:
                        names_ids[name] = current_id
                        current_id +=1
                    
                    id = names_ids[name]
 
                    gray_image = cv.imread(file_path,cv.IMREAD_GRAYSCALE)                                   # convert image to grayscale
                    #cv.imshow("Faces on image", gray_image)
                    #cv.waitKey(0)                                                                         
                    gray_image_array = np.array(gray_image, "uint8")                                        # convert grayscale image to np array
                    faces = face_cascade.detectMultiScale(gray_image_array, 1.12, 5)                        # find human faces on img
                    if len(faces) > 1 or len(faces) == 0: print(file," nie ma twarzy")
                   
                    
                    for (x, y, w, h) in faces:
                        roi_gray_image = gray_image[y:y+h, x:x+w]                                           # extract area in image where face appears
                        faces_to_train.append(roi_gray_image)
                        ids.append(id)
                        cv.rectangle(gray_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    cv.destroyAllWindows()
                   
                    
                #change_names(dp, name_biggest_index, files, dirname)
                print(names_ids)
                print(ids)
                


with open("names.pickle", 'wb') as f:
    pkl.dump(names_ids, f)
    f.close()

face_recognizer.train(faces_to_train, np.array(ids))
face_recognizer.save("trainner.yml")


