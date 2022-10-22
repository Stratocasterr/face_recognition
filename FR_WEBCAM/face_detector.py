import cv2 as cv
import face_recognition as fr
import numpy as np
from sklearn import svm
import os

# by me:
encodes_of_known = []
names_of_known = []

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
                    #if face_location: face_location = face_location[0]
                    #else: print("There is no face in: ", file)

                    #face_encode = fr.face_encodings(image_file)[0]
                    #encodes_of_known.append(face_encode)

                    print("Encoding images in: ", dirname, "dictionary. (" ,index + 1, "/",len(files), ") total: ", len(encodes_of_known))
                print("\n")

    
# find people
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



''' 
# by github 
def encode_known_people():

    # Training the SVC classifier

    # The training data would be all the face encodings from all the known images and the labels are their names
    encodings = []
    names = []

    # Training directory
    train_dir = os.listdir('small_learn_images/')

    # Loop through each person in the training directory
    for person in train_dir:
        pix = os.listdir("small_learn_images/" + person)

        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = fr.load_image_file("small_learn_images/" + person + "/" + person_img)
            face_bounding_boxes = fr.face_locations(face)

            #If training image contains exactly one face
            if len(face_bounding_boxes) == 1:
                face_enc = fr.face_encodings(face)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
            else:
                print(person + "/" + person_img + " was skipped and can't be used for training")

    # Create and train the SVC classifier
    clf = svm.SVC(gamma='scale')
    clf.fit(encodings,names)




    for dirpath, dirnames, filenames in os.walk('test_on_collage/'):
        for filename in filenames:
            faces = []

            # Load the test image with unknown faces into a numpy array
            test_image = fr.load_image_file('test_on_collage/' + str(filename))

            # Find all the faces in the test image using the default HOG-based model
            face_locations = fr.face_locations(test_image)
            no = len(face_locations)
            print("Number of faces detected: ", no)

            # Predict all the faces in the test image using the trained classifier
            print("Found:")
            for i in range(no):
                test_image_enc = fr.face_encodings(test_image)[i]
                name = clf.predict([test_image_enc])
                print(*name)


'''