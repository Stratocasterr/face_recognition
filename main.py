from face_detector import *
from find_faces import *

base_folder = "small_learn_images/"
learn_images_path = r'C:\Users\kacpe\Desktop\face_recognition\learn_images/'
test_images_path = r'C:\Users\kacpe\Desktop\face_recognition\test_on_collage'


#            ***Test if there are faces in your dataset images***

# arg1 = folder you want to test , arg2 = put true if you want to see located faces on all images 



#            ***Find people in your images***


# encode_known_people(base_folder, learn_images_path)

# find_people_on_images(test_images_path)


#test_dataset()

encode_known_people(base_folder, learn_images_path)

find_people_on_images(test_images_path, True)