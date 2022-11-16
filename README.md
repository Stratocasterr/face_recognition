# Face recognition with Python


This program is a part of my Bachelor degree project


## Recognize famous people faces on images
In this part you can recognize famous people on given image. 


## Recognize faces in real-time using webcam

### How to use it?

#### install requirements
1. openCV  ( I used 4.6.0.66)
2. numpy (I used v1.23.3)
#### Add your database.

--DATABASE DIRECTORY
        --person1_name
                --person1_1.img
                --person1_2.img
                --person1_3.img
                ...
                
        --person2_name
                --person2_1.img
                --person2_2.img
                --person2_3.img
                ...
                
         --person3_name
                --person3_1.img
                --person3_2.img
                --person3_3.img
                ...
            
 put your database here:
 

 
#### Change paths in config.py file

PROJECT_DICT_PATH        -> path to directory, where project is located  
IMAGES_FOR_TRAIN_PATH    -> path to directory, where your database is located  

#### run main.py 


To realize my project, I used OpenCV library. CascadeClassifier for detecting faces in image and LBPHFaceRecognizer to recognize specified person.

Recognizer is based on LBPH Algorithm. 
One may read about it here: https://towardsdatascience.com/face-recognition-how-lbph-works-90ec258c3d6b

I tested it with my friend Karol.
