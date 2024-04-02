import cv2
import numpy as np
import findGenderAndAge

#load the haar cascade classifiers for face and eye detection
face_detector = cv2.CascadeClassifier("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml")
eye_detector = cv2.CascadeClassifier("C:\\opencv\\build\etc\\haarcascades\\haarcascade_eye.xml")

#open video capture, if the arguement is 0, then it will open computer camera
video_capture = cv2.VideoCapture(0)

#initialize the shortest distance of two points, set the beggining value as the infinity
shortest_distance = float("inf")

#processing inside this while loop
while True:

    #read a clip from the video
    ret, original = video_capture.read()

    # get the middle point of the whole video
    original_width, original_height, _ = original.shape
    midPoint = (original_height//2, original_width//2)

    #break the loop if no clip is retrieved
    if not ret:
        break

    #convert the image to grayscale (gray is good for detection)
    gray_img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)#color from BGR to Gray

    #enhance the gray color contrast for better detection
    gray_img = cv2.equalizeHist(gray_img)

    #results lists coordinates (x,y,w,h) of bounding boxes around the detected object
    #perform face detection
    results = face_detector.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (x1,y1,w1,h1) in results:
        #draws a green rectangle around each detected face and creates regions of interest(ROI) 
        img = cv2.rectangle(original,(x1, y1),(x1 + w1, y1 + h1),(0,255,0))#(x,y) are the coordinates of top-left corner of rectangle, obtained by detected face. (x+w, y+h) is the coordinates of the bottom-right corner of the rectangle from detected face
        
        #import gender detection function
        findGenderAndAge.findGender(img, original)

        head_middle = (x1+(w1//2), y1+(h1//2))#the coordinate of middle of the head
        cv2.circle(original, head_middle, 5, (0,0,255), -1)#then draw the middle of the head
        cv2.line(original, head_middle, midPoint, (0,0,255), 1)#connect the line between head_middle and midPoint

        #then calculate the distance between these two points
        shortest_distance = np.linalg.norm(np.array(head_middle) - np.array(midPoint))
        

    #extract regions of interest (ROI) from the grayscale image (gray_img) and the original color image (img) based on the coordinates of the detected faces.
    roi_gray = gray_img[y1 : y1 + h1, x1 : x1 + w1]
    roi_color = img[y1 : y1 + h1, x1 : x1 + w1]

    #for eye detection, detects eyes within each face region and draws rectangles around them
    eyes = eye_detector.detectMultiScale(roi_gray)
    for (x2, y2, w2, h2) in eyes:
        cv2.rectangle(roi_color,(x2, y2),(x2 + w2, y2 + h2),(0,255,0))

        #measure the area of eye
        eye_area = w2*h2


    text = "Head Position Index: " + str(shortest_distance)
    cv2.putText(original,text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

    text = "Eye Area: " + str(eye_area)
    cv2.putText(original,text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
    

    #draw the middel point of whole video:
    cv2.circle(original, midPoint, 5, (0,0,255), -1)

    #show the result
    cv2.imshow('output',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release video
video_capture.release()
cv2.destroyAllWindows()