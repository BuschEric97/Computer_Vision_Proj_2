# Computer_Vision_Proj_2

In this project you are asked to write two programs to detect “winking”. In writing your
programs you may use all of the high level functionality of OpenCV, but you are not allowed to
convert images into arrays of pixel values. The only cascade classifiers that can be used are those
provided by OpenCV, as well as those available in the following link:
http://alereimondo.no-ip.org/OpenCV/34.
The cascade classifiers provided by OpenCV can be found in:
https://github.com/opencv/opencv/tree/master/data/haarcascades.
They are also available as part of the OpenCV distribution.

The input to each program is a folder containing images or a live video feed. The program
displays each image, and marks each detected face with a distinct color. It also computes and
prints the total number of detections.


First program: DetectWink1.py
Write an OpenCV program that can detect a winking face. You may want to build your program
by changing the example program DetectWink.py.


Second program: DetectWink2.py
The requirements for the second program are the same as the requirements for the first program,
except that it must start by applying a filter to the image. The filter can be histogram equalization,
smoothing, or anything else that you may consider to be useful.

Suggestion of changes
1. Changing the parameters of the functions
2. Including or changing different preprocessing steps
1
3. Changing region of interest
4. Changing the logic of the program.
5. Anything else which would improve the correct detection and reduce the incorrect detections.
