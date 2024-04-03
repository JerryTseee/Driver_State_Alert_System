import cv2
import numpy as np
import findGenderAndAge
import time


#the function is used to detect the danger driving
#when these parameters under certain range, then it is safe, or it is dangerous
def check(original, shortest_distance, eye_area, age):

    output = "Driver State: Safe Driving"

    #initialize the close eye numbers
    close_eye_number = 0

    if shortest_distance > 150:
        output = "Driver State: Danger Driving!!!"
    if eye_area < 600:
        output = "Driver State: Danger Driving!!!"
    if int(age) < 20:
        output = "Driver State: Danger Driving!!!"

    cv2.putText(original, output, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
    return output