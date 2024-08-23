import cv2
import os
import numpy as np

# Import Python Image Library (PIL)
from PIL import Image

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Using prebuilt frontal face training model, for face detection
detector = cv2.CascadeClassifier("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml")

def sort_key(file_path):
    parts = os.path.basename(file_path).split('.')
    if len(parts) >= 3:
        return int(parts[2])
    return 0

# Create method to get the images and label data
def getImagesAndLabels(path):

    # Get all file path
    imagePaths = [os.path.join(path,i) for i in os.listdir(path)]
    imagePaths = sorted(imagePaths, key=sort_key)

    for i in imagePaths:
        print(i)
    
    # Initialize empty face sample
    faceSamples=[]
    
    # Initialize empty id
    ids = []

    # Loop all the file path
    for imagePath in imagePaths:

        # Get the image and convert it to grayscale
        PIL_img = Image.open(imagePath).convert('L')

        # PIL image to numpy array
        img_numpy = np.array(PIL_img,'uint8')
        print(img_numpy)

        # Get the image id
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        print(id)

        # Get the face from the training images
        faces = detector.detectMultiScale(img_numpy)
        print(faces)

        # Loop for each face, append to their respective ID
        for (x,y,w,h) in faces:

            print(x,y,w,h)

            # Add the image to face samples
            faceSamples.append(img_numpy[y:y+h,x:x+w])

            # Add the ID to IDs
            ids.append(id)

    # Pass the face array and IDs array
    print(faceSamples)
    print(ids)
    return faceSamples,ids

# Get the faces and IDs
faces,ids = getImagesAndLabels("D:\\DriverAlertSystem\\face_identity\\dataset")

# Train the model using the faces and IDs
flag = np.array(ids)
print(faces)
print(flag)
recognizer.train(faces, flag)

# Save the model into trainer.yml
recognizer.save('D:\\DriverAlertSystem\\face_identity\\trainer.yml')
