# Face recognition with Python

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
One may read about it in [article](https://towardsdatascience.com/face-recognition-how-lbph-works-90ec258c3d6b).

## Demo with my friends


https://user-images.githubusercontent.com/101999487/202310284-edd94465-32c3-42af-b9b0-bf194eb813d4.mp4



https://user-images.githubusercontent.com/101999487/202311117-0959ad3e-f9cb-4d1d-80fe-cf2e91483747.mp4


