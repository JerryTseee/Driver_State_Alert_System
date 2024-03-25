import cv2
import numpy as np

#load the image
image = cv2.imread("C:\\Users\\admin\Pictures\\WhatsApp Image 2023-01-07 at 11.18.13.jpg")

#convert the image to grayscale (gray is good for detection)
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#color from BGR to Gray

#load the haar cascade classifiers for face and eye detection
face_detector = cv2.CascadeClassifier("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml")
eye_detector = cv2.CascadeClassifier("C:\\opencv\\build\etc\\haarcascades\\haarcascade_eye.xml")



#results lists coordinates (x,y,w,h) of bounding boxes around the detected object
#perform face detection
results = face_detector.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
for (x,y,w,h) in results:
    #draws a green rectangle around each detected face and creates regions of interest(ROI) 
    img = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0))#(x,y) are the coordinates of top-left corner of rectangle, obtained by detected face. (x+w, y+h) is the coordinates of the bottom-right corner of the rectangle from detected face

#extract regions of interest (ROI) from the grayscale image (gray_img) and the original color image (img) based on the coordinates of the detected faces.
roi_gray = gray_img[y:y+h, x:x+w]
roi_color = img[y:y+h, x:x+w]

#for eye detection, detects eyes within each face region and draws rectangles around them
eyes = eye_detector.detectMultiScale(roi_gray)
for (x,y,w,h) in eyes:
    cv2.rectangle(roi_color,(x,y),(x+w,y+h),(0,255,0))


#show the result
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()