# Driver State Alert System  
real time driver state detection, it is implemented with Python and OpenCV.  
It can be used to track the safety of the driver and induce the alert system when there are dangerous behaviors  
# Demo  
<b>Video Demonstration:  
https://github.com/JerryTseee/Driver_State_Alert_System/assets/126223772/6a0484c1-a990-471b-aa0a-958a244fff80  
  
<b>Pictures Demonstration:  
<img width="437" alt="image" src="https://github.com/JerryTseee/Driver_State_Alert_System/assets/126223772/20e059e7-60dc-414d-9d6e-50f0cb10c3a7">
<img width="326" alt="image" src="https://github.com/JerryTseee/FaceEyeDetection/assets/126223772/058d421a-457b-4b2e-88ae-0a831e645c5b">  
# How Does It Work  
The system is implemented using Python with OpenCV, the Haar Cascades algorithm is used in this project.
The system captures the stream from the computer camera, then the captured images will be converted to the gray images for better detection.
After the above preparation, system will load the haar cascade classifiers for the detection of faces and eyes, then it will draw the rectangles
around the detected faces and eyes. Pre-trained gender model and age model will be loaded into the main function for the gender recognition and
age recognition. Finally, the check function will analyze all the obtained data and output the alertion result, when the specific data exceeds
certain range, it will output the positive alert.

  
(relevant metrics need to be adjusted according to the real world situations)  
- if the Head Position Index exceeds certain range, it is an inappropriate driving (dangerous)  
- if the age of driver is below 20, it is an illegal driving (dangerous)  
- if the Eye Close Number exceeds certain range in a period, it is a tired driving (dangerous)  
- Head Position Index measures the distance between the mid point of the camera and the mid point of the detected face  
- Eye Close Number is obtained by counting the number of times when the detected eyes area is smaller than certain range, this number will be reset to zero after open the eyes  
