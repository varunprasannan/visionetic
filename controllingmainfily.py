from imutils.video import VideoStream
import numpy as np

import cv2
import imutils
import time

# importing directkeys for using press key and release key functions
from directkeys import  W, A, S, D
from directkeys import PressKey, ReleaseKey 

# define the lower and upper boundaries of the "blue" object in the HSV color space
# https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv
blueLower = np.array([53, 187, 0])
blueUpper = np.array([180, 255, 255])

# Staring the webcam
video = VideoStream(src=0).start()
 
# allow the camera or video file to warm up
time.sleep(2.0)
initial = True
flag = False
current_key_pressed = set()
circle_radius = 30

# Defining window boundaries for each logically divided region
windowSize = 80
windowSize2 = 80

lr_counter = 0

# keep looping
while True:
    keyPressed = False
    keyPressed_lr = False
    
    # grab the current frame
    frame = video.read()
    
    # My video frame was flipped horizontally. If your video is not flipped by default you can ommit this
    frame = cv2.flip(frame,1);

    # resize the frame, blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=600)
    frame = imutils.resize(frame, height=300)

    # storing height and width in varibles
    height = frame.shape[0]
    width = frame.shape[1]
    
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # create a mask for the blue color and perform opening and closing to remove any small
    # blobs left in the mask 
    mask = cv2.inRange(hsv, blueLower, blueUpper)
    kernel = np.ones((5, 5), np.uint8)
    # inspired by https://pythonprogramming.net/morphological-transformation-python-opencv-tutorial/
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    up_mask = mask[0:height//2,0:width,]
    down_mask = mask[height//2:height,width//4:3*width//4,]

    # find the contours(blue object's boundary) in the left and right frame to find the center of the object
    # syntax: (img,mode,method)
    cnts_up = cv2.findContours(up_mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts_up = imutils.grab_contours(cnts_up)
    center_up = None

    cnts_down = cv2.findContours(down_mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts_down = imutils.grab_contours(cnts_down)
    center_down = None
 
    # only proceed if at least one contour was found
    if len(cnts_up) > 0:
       
        c = max(cnts_up, key=cv2.contourArea)
 
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        
        M = cv2.moments(c)
        # print(M)
        
        center_up = (int(M["m10"] / (M["m00"]+0.000001)), int(M["m01"] / (M["m00"]+0.000001)))
    
        # only proceed if the radius meets a minimum size
        if radius > circle_radius:
            # draw the circle and centroid on the frame,
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center_up, 5, (0, 0, 255), -1)
            if center_up[0] < (width // 2 - windowSize // 2):
                # cv2.putText(frame,'LEFT',(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
                PressKey(A)
                print("A")
                current_key_pressed.add(A)
                keyPressed = True
                keyPressed_lr = True
            elif center_up[0] > (width // 2 + windowSize // 2):
                # cv2.putText(frame,'RIGHT',(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
                PressKey(D)
                print("D")
                current_key_pressed.add(D)
                keyPressed = True
                keyPressed_lr = True
    
    # only proceed if at least one contour was found
    if len(cnts_down) > 0:
        c2 = max(cnts_down, key=cv2.contourArea)
        ((x2, y2), radius2) = cv2.minEnclosingCircle(c2)
        M2 = cv2.moments(c2)
        # print(M2)
        center_down = (int(M2["m10"] / (M2["m00"]+0.000001)), int(M2["m01"] / (M2["m00"]+0.000001)))
        center_down = (center_down[0]+width//4,center_down[1]+height//2)
    
        # only proceed if the radius meets a minimum size
        if radius2 > circle_radius:
            # draw the circle and centroid on the frame,
            cv2.circle(frame, (int(x2)+width//4, int(y2)+height//2), int(radius2),
                (0, 255, 255), 2)
            cv2.circle(frame, center_down, 5, (0, 0, 255), -1)
            
            if (height//2) < center_down[1] < ((4*height)//4) and (0*width) < center_down[0] < ((2*width)//4):
                # cv2.putText(frame,'UP',(200,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
                PressKey(W)
                print("W")
                keyPressed = True
                current_key_pressed.add(W)
            elif (height//2) <center_down[1] < ((4*height)//4 ) and (width//2) < center_down[0] < ((4*width)//4):
                # cv2.putText(frame,'DOWN',(200,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
                PressKey(S)
                print("S")
                keyPressed = True
                current_key_pressed.add(S)

   
    frame_copy = frame.copy()
    
    # draw box for left (A)
    frame_copy = cv2.rectangle(frame_copy, (0, 0), (width//2 - windowSize//2, height//2), (250, 237, 39), 1)
    cv2.putText(frame_copy,'LEFT',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))

    # draw box for left (D)
    frame_copy = cv2.rectangle(frame_copy,(width//2 + windowSize//2,0),(width-2,height//2 ),(0,255,255),1)
    cv2.putText(frame_copy,'RIGHT',(438,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))

    # draw box for left (W)
    frame_copy = cv2.rectangle(frame_copy,(2*width//10,(height//2)+5),(width//2,4*height//4),(39,250,237),1)
    cv2.putText(frame_copy,'UP',(width//4,(height//2)+33),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))

    # draw box for left (S)
    frame_copy = cv2.rectangle(frame_copy,(width//2,(height//2)+5),(8*width//10,height),(39,237,250),1)
    cv2.putText(frame_copy,'DOWN',((3*width//4)-100,(height//2)+108),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))

    # display final frame
    cv2.imshow("Frame", frame_copy)

    if not keyPressed and len(current_key_pressed) != 0:
        for key in current_key_pressed:
            ReleaseKey(key)
        current_key_pressed = set()

    if not keyPressed_lr and ((A in current_key_pressed) or (D in current_key_pressed)):
        if A in current_key_pressed:
            ReleaseKey(A)
            current_key_pressed.remove(A)
        elif D in current_key_pressed:
            ReleaseKey(D)
            current_key_pressed.remove(D)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
 

video.stop()
# close all windows
cv2.destroyAllWindows()
