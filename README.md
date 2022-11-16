# Face recognition with Python


This program is a part of my Bachelor degree project


## Recognize famous people faces on images
In this part you can recognize famous people on given image. 


## Recognize faces in real-time using webcam
Here one may recognize any person, only by uploading their photos to the database.
### How to use it?

#### 1. install requirements
1. openCV  ( I used 4.6.0.66)
2. numpy (I used v1.23.3)


#### 2. add your database.

Database directory should contain photos of people one want to recognize. These photos need to be taken from different angles, distances, using different lighting and background. The more images of each person one upload, the better face recognize results will be. One have to remember, that each person's directory should contain similar amount of images.

database directory structure:

```bash
YOUR_DATABASE_DIR
              ├── person1_name
              │   ├── person1_1.img
              │   ├── person1_2.img
              │   └── person1_3.img
                  ...
              ├── person1_name
              │   ├── person1_1.img
              │   ├── person1_2.img
              │   └── person1_3.img
                   ...
              ...
```

            
Put your database in "face_recognition_webcam" directory.
 

 
#### 3. change paths in config.py file

PROJECT_DICT_PATH        -> path to directory, where project is located  
IMAGES_FOR_TRAIN_PATH    -> path to directory, where your database is located  

#### 4. run main.py 

If you have webcam in your device, face recognizer window will appear. One can specify camera device by changing video parameter in "faces_detection.py" file.

```bash
video = cv.VideoCapture(0)    // 0 is default value here
```


To realize my project, I used OpenCV library. CascadeClassifier for detecting faces in image and LBPHFaceRecognizer to recognize specified person.

Recognizer is based on LBPH Algorithm. 
One may read about it here: https://towardsdatascience.com/face-recognition-how-lbph-works-90ec258c3d6b

I tested it with my friend Karol.
