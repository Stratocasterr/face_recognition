from face_detector import *
from find_faces import *
from webcam import *

learn_images_path = "friends_images/"
test_images_path = "generated_collages/"


#            ***Test if there are faces in your dataset images***

# arg1 = folder you want to test , arg2 = put true if you want to see located faces on all images 



#            ***Find people in your images***


# encode_known_people(base_folder, learn_images_path)

# find_people_on_images(test_images_path)


#test_dataset()

#encode_known_people(learn_images_path)
#recognize_people_on_webcam()
#find_people_on_images(test_images_path, True)
recognize_people_on_webcam()