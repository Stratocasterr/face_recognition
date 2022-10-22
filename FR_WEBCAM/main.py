from turtle import width
import numpy as np
import cv2 as cv

filename = 'video.avi' 
cap = cv.VideoCapture(0)
run = True

def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

def rescale_frame(frame, percent=75):
    scale_percent = 75
    height = int(frame.shape[0] * scale_percent / 100)
    width = int(frame.shape[1] * scale_percent / 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation = cv.INTER_AREA)


while run:
    ret, frame = cap.read()
    cv.imshow('frame', frame)

    if cv.waitKey(20) & 0xFF == ord('q'):
        run = False

cap.release()
cv.destroyAllWindows()

