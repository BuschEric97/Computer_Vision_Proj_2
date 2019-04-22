# CS4391.001 - Project 2
#
# This program employs the same strategies
# DetectWink1.py with the addition of
# frame preprocessing
#
# Program by: Eric Busch
# edb160230@utdallas.edu

import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys

def detectWink(frame, location, ROI, cascade):
    #ROI = cv2.equalizeHist(ROI)
    #ROI = cv2.medianBlur(ROI, 3)

    # run cascade detection algorithm for eyes
    eyes = cascade.detectMultiScale(
        ROI, 1.15, 4, 0|cv2.CASCADE_SCALE_IMAGE, (5, 10))

    # iterate through each eye detected in order to put a box around it
    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2) # put a red box around the detected eye

    # return whether or not a winking face was detected (one eye is detected)
    return len(eyes) == 1

def detect(frame, faceCascade, eyesCascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert the current frame to greyscale

    # frame preprocessing
    gray_frame = cv2.equalizeHist(gray_frame)
    #gray_frame = cv2.medianBlur(gray_frame, 3)

    # run cascade detection algorithm for faces
    scaleFactor = 1.15 # range is from 1 to ..
    minNeighbors = 3   # range is from 0 to ..
    flag = 0|cv2.CASCADE_SCALE_IMAGE # either 0 or 0|cv2.CASCADE_SCALE_IMAGE
    minSize = (30,30) # range is from (0,0) to ..
    faces = faceCascade.detectMultiScale(
        gray_frame,
        scaleFactor,
        minNeighbors,
        flag,
        minSize)

    # count the number of detected faces
    detected = 0
    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3]
        faceROI = gray_frame[y:int(y+(h*0.6)), x:x+w] # Set faceROI to be the top 60% of the face because eyes are only ever in that area
        if detectWink(frame, (x, y), faceROI, eyesCascade):
            detected += 1 # increment detected if a winking face was detected
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2) # put a blue box around the detected winking face
        else:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2) # put a green box around the detected face
    return detected


def run_on_folder(cascade1, cascade2, folder):
    if(folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder,f) for f in listdir(folder) if isfile(join(folder,f))]

    # go through each image in the specified folder and detect faces
    windowName = None
    totalCount = 0
    for f in files:
        img = cv2.imread(f)
        if type(img) is np.ndarray:
            lCnt = detect(img, cascade1, cascade2) # run face detection algorithms
            totalCount += lCnt
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return totalCount

def runonVideo(face_cascade, eyes_cascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    # process live video feed for face detection
    windowName = "Live Video"
    showlive = True
    while(showlive):
        ret, frame = videocapture.read() # get next frame

        if not ret:
            print("Can't capture frame")
            exit()

        detect(frame, face_cascade, eyes_cascade) # run face detection algorithms
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showlive = False

    # release video capture
    videocapture.release()
    cv2.destroyAllWindows()

# "main" routine
if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + len(sys.argv) - 1
              + "arguments. Expecting 0 or 1:[image-folder]")
        exit()

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                      + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                      + 'haarcascade_eye.xml')

    # determine whether to run on live video feed or folder of images
    if(len(sys.argv) == 2): # one argument
        folderName = sys.argv[1]
        detections = run_on_folder(face_cascade, eye_cascade, folderName)
        print("Total of ", detections, "detections")
    else: # no arguments
        runonVideo(face_cascade, eye_cascade)
