"""
using this file to conduct age recognition and gender recognition
"""
import cv2
import numpy as np

def findGender(img, original):
    #load the pre-trained models for gender recognition
    gender_net = cv2.dnn.readNetFromCaffe("D:\\DriverAlertSystem\\gender_deploy.prototxt", "D:\\DriverAlertSystem\\gender_net.caffemodel")
    
    #preprocess the face image for gender recognition
    blob = cv2.dnn.blobFromImage(img, 1.0, (227,227), (78.426, 87.767, 114.897), swapRB = False)

    #perform gender recognition
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender_idx = np.argmax(gender_preds)
    if gender_idx == 1:
        gender = "Female"
    else:
        gender = "Male"
    

    #show
    text = "Gender: " + str(gender)
    cv2.putText(original, text, (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
    return gender

#same logic for the findAge function
def findAge(img, original):
    age_net = cv2.dnn.readNetFromCaffe("D:\\DriverAlertSystem\\age_deploy.prototxt", "D:\\DriverAlertSystem\\age_net.caffemodel")
    blob = cv2.dnn.blobFromImage(img, 1.0, (227,227), (78.426, 87.767, 114.897), swapRB = False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age_idx = np.argmax(age_preds)
    age = str(age_idx+20) # the model is not that accurate, always 4-8 years old, so I manually add back 20

    text = "Age: " + age
    cv2.putText(original, text, (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
    return age