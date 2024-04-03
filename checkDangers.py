import cv2
import numpy as np
import findGenderAndAge
import time



#the function is used to detect the danger driving
#when these parameters under certain range, then it is safe, or it is dangerous
def check(original, shortest_distance, eye_area, age, eye_close_times):

    output = "Driver State: Safe Driving"   

    #if head position is not correct -> danger
    if shortest_distance > 150:
        output = "Driver State: Danger Driving!!!"

    #if you close your eye frequently -> danger
    if eye_area < 600:
        eye_close_times += 1
    else:
        #once you open your eye again, it will clear to zero and restart the counting
        eye_close_times = 0
    if eye_close_times > 15:
        output = "Driver State: Danger Driving!!!"

    #if your age is under 20 -> danger
    if int(age) < 20:
        output = "Driver State: Danger Driving!!!"


    #print the result
    text = "Eye Close Times: "+str(eye_close_times)
    cv2.putText(original, text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
    cv2.putText(original, output, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
    return eye_close_times