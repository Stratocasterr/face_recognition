import cv2 as cv
import face_recognition as fr
import numpy as np
from sklearn import svm
import os

# by me:
encodes_of_known = []
names_of_known = []
learn_images_path =
# learning
def encode_known_people(learn_images_path):
   
    for dirpath, dirnames, filenames in os.walk(learn_images_path):
       
        for dirname in dirnames:
            for dp, dirs, files in os.walk(learn_images_path+ str(dirname)):
                print(dirname)
                for index, file in enumerate(files):
                    
                    file_path = os.path.join(dp, file)
                    name = dp.split("/")[1]
                    names_of_known.append(name)
                    image_file = cv.imread(file_path)
                    face_location = fr.face_locations(image_file)
                    print(face_location)
                    if face_location: face_location = face_location[0]
                    else: print("There is no face in: ", file)

                    face_encode = fr.face_encodings(image_file)[0]
                    encodes_of_known.append(face_encode)

                    print("Encoding images in: ", dirname, "dictionary. (" ,index + 1, "/",len(files), ") total: ", len(encodes_of_known))
                print("\n")

    
# find people on images
def find_people_on_images(test_images_path, show_images = False):

    for dirpath, dirnames, filenames in os.walk(test_images_path):

        for filename in filenames:
            faces = []
            
            file_path = os.path.join(dirpath, filename)
            image_file = cv.imread(file_path)
            face_encode = fr.face_encodings(image_file)

            face_locations = fr.face_locations(image_file)
            
            if face_locations:
                print("Number of faces in that photo: ", len(face_locations))
                for face in face_locations: faces.append(face)

            else: print("There is no face in: ", filename)
        
            for index,face in enumerate(faces):
                actual_face_encode = face_encode[index]
                matches = []
                count_matches = {}
                final_choice = "Unknown"
                matches = [[fr.compare_faces([known], actual_face_encode), index] for index, known in enumerate(encodes_of_known)]
               
                true_indexes = [match[1] for match in matches if match[0][0]]

                for i in true_indexes:
                    name = names_of_known[i]
                    count_matches[name] = count_matches.get(name, 0) + 1

                correctness_index = 0
                for key in count_matches.keys():
                    correctness_index += count_matches[key]

                
                if count_matches:
                    final_choice = max(count_matches, key=count_matches.get)

                    correctness_index = (count_matches[final_choice] / correctness_index) *100                          # in % of how many times is similar with a few people's photos
                    similarity_index =  (count_matches[final_choice]  / names_of_known.count(final_choice)) *100        # in % of how many times is similar with only one person's photos
                    if similarity_index < 20: final_choice = "Unknown"

                else:
                    
                    correctness_index = 0
                    similarity_index = 0

                print("this is:", final_choice," (", correctness_index, "(%)) compare to other's images, and ", similarity_index, "(%) compare to itself's images")

                y_1, x_2, y_2, x_1 = face
                cv.rectangle(image_file, (x_1, y_1), (x_2, y_2), (0, 0, 255), 2)
                cv.putText(image_file, f'This is: , {final_choice}', (x_1-20, y_2-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
            if show_images:
                cv.namedWindow("Face Recognition", cv.WINDOW_NORMAL)
                cv.resizeWindow("Face Recognition", 1280, 960)
                cv.imshow("Face Recognition",image_file)
                cv.waitKey(0)


# find people on webcam
def find_people_on_webcam():
    video = cv.VideoCapture(0)                                                                                                 # capture video cam frames and set parameters
    video.set(cv.CAP_PROP_FRAME_WIDTH, 1920)                                                                                   
    video.set(cv.CAP_PROP_FRAME_HEIGHT, 1440)

    running = True
    while running:
        faces = []
        ret, color_frame = video.read()
        gray_frame = cv.cvtColor(color_frame, cv.COLOR_BGR2GRAY)                                                          # convert video frame to grayscale
        face_encode = fr.face_encodings(gray_frame)
        face_locations = fr.face_locations(gray_frame)

        if face_locations:
                for face in face_locations: faces.append(face)


        for index,face in enumerate(faces):
            actual_face_encode = face_encode[index]
            matches = []
            count_matches = {}
            final_choice = "Unknown"
            matches = [[fr.compare_faces([known], actual_face_encode), index] for index, known in enumerate(encodes_of_known)]
            
            true_indexes = [match[1] for match in matches if match[0][0]]

            for i in true_indexes:
                name = names_of_known[i]
                count_matches[name] = count_matches.get(name, 0) + 1

            correctness_index = 0
            for key in count_matches.keys():
                correctness_index += count_matches[key]

            
            if count_matches:
                final_choice = max(count_matches, key=count_matches.get)

                correctness_index = (count_matches[final_choice] / correctness_index) *100                          # in % of how many times is similar with a few people's photos
                similarity_index =  (count_matches[final_choice]  / names_of_known.count(final_choice)) *100        # in % of how many times is similar with only one person's photos
                if similarity_index < 20: final_choice = "Unknown"

            else:
                
                correctness_index = 0
                similarity_index = 0

            
            print(face)
            print(face[0])
            ''' 
            y_1, x_2, y_2, x_1 = face
            cv.rectangle(color_frame, (x_1, y_1), (x_2, y_2), (0, 0, 255), 2)
            cv.putText(color_frame, f'{final_choice}', (x_1-20, y_2-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.imshow("Face Recognition",color_frame)
            '''