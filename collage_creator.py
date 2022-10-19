from os import listdir
from random import random
import cv2 as cv
import numpy as np
import random

def create_collage(collage_size, size, from_dir, dest_dir, random_images, *images):

    if random_images: 
        images = [image for image in images]
        print(images)
    else: 
        given_dir_images = listdir(from_dir)
        given_dir_images_len = len(given_dir_images)

        images = [from_dir + "/" + given_dir_images[random.randint(0, given_dir_images_len - 1)] 
            for index in range(collage_size[0] * collage_size[1])]
        print(images)


    # create image objects and resize
    single_img_x_size = size[0] // collage_size[0]
    single_img_y_size = size[1] // collage_size[1]
    single_img_size = (single_img_x_size, single_img_y_size)
    columns = []
    actual_index = 0


    images = [cv.resize(cv.imread(image), single_img_size) for image in images]

    for horizontal in range(collage_size[0]):
        column = np.vstack(images[actual_index + index] for index in range(collage_size[1]))
        columns.append(column)
        actual_index += collage_size[0]

    collage = columns[0]
    for column in (columns[1:]):
        collage = np.hstack([collage, column])
    
    cv.imshow("your collage", collage)
    cv.waitKey(0)
    collage_index = "0"

    print(listdir(dest_dir))
    while "your_collage" + collage_index + ".jpg" in listdir(dest_dir): 
        collage_index = str(random.randint(0,100))
        print(collage_index)
    cv.imwrite(dest_dir + "/your_collage" + collage_index + ".jpg", collage)

   
create_collage((2, 2), (640, 480), 'test_images', 'generated_collages', False)