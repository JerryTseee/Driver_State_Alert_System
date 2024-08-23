import cv2

# Start capturing video 
vid_capture = cv2.VideoCapture(0)

# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml")

# For each person, one face id
face_id = 5

# Initialize sample face image
count = 0

# Start looping
while(True):

    # Capture video frame
    _, image_frame = vid_capture.read()

    # Convert frame to grayscale color
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Loops for each faces
    for (x,y,w,h) in faces:

        # Crop the image frame into rectangle
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
        # Increment sample face image
        count += 1

        # Save the captured image into the datasets folder: this folder contains the 100 images of the person for training
        cv2.imwrite("D:\\DriverAlertSystem\\face_identity\\dataset\\User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', image_frame)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # If image taken reach 100, stop taking video (we only want 100 images)
    elif count>100:
        break

# Stop video
vid_capture.release()

# Close all started windows
cv2.destroyAllWindows()
