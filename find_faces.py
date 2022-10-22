import cv2 as cv
import face_recognition as fr
import numpy as np
import os



base_folder = "friends_images/"

def test_dataset(dataset = "learn_images/", show_faces_on_images = False):
    if base_folder not in dataset: dataset = base_folder + dataset
    for dirpath, dirnames, filenames in os.walk(dataset):

        if not dirnames:
            detect_faces_in_files(dataset.split('/')[1], dirpath.split('/')[0], show_faces_on_images)
        else: 
            
            for dirname in dirnames:
                detect_faces_in_files(dirname, dirpath, show_faces_on_images)
   
def detect_faces_in_files(dirname, dirpath, show_faces_on_images):
    for dp, dirs, files in os.walk(base_folder+ str(dirname)):
        print(dirname)
        face_locations = []
        for file in files:
        
            file_path = os.path.join(dirpath,dirname, file)
            image_file = cv.imread(file_path)
            face_location = fr.face_locations(image_file)

            if not len(face_location): print("There is no face in: ", file)
            else:
                for fl in face_location:
                    
                    
                    y_1, x_2, y_2, x_1 = fl
                    cv.rectangle(image_file, (x_1, y_1), (x_2, y_2), (0, 0, 255), 2)

                    print("In file: ", file, " found face in: ", fl)
                    face_locations.append(fl)
            
            if show_faces_on_images: 
                y_1, x_2, y_2, x_1 = face_location[0]
                cv.rectangle(image_file, (x_1, y_1), (x_2, y_2), (0, 0, 255), 2)

            if show_faces_on_images: 
                cv.imshow(file,image_file)
                cv.waitKey(0)

        print("In: ", dirname, " found: ", len(face_locations), " faces in: ", len(files), " images.", "\n")




