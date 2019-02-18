#!/usr/bin/env python

import cv2
import numpy as np 

def draw_marker(img, circles):
    # img = cv2.imread(img,0)
    borderimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for i in circles[0,:]:
    # draw the outer circle
        cv2.circle(borderimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(borderimg,(i[0],i[1]),2,(0,0,255),3)
        cv2.putText(borderimg,str(i[0])+str(',')+str(i[1]), (i[0],i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
    return borderimg

def detect_circular_objects(image_path):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gray_blur = cv2.medianBlur(gray, 13)  # Remove noise 
    gray_lap = cv2.Laplacian(gray_blur, cv2.CV_8UC1, ksize=5) # Laplacian on image to remove the granularity

    # kernel = np.ones((5,5), np.uint8) 
    # dilate_lap = cv2.dilate(gray_lap, kernel, iterations=7)   # Fill in gaps from blurring. This helps to detect circles with broken edges.
    # Image = cv2.cvtColor(dilate_lap, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("output.png", dilate_lap)
    # Remove in space between the groups of circles.
    # lap_blur = cv2.bilateralFilter(dilate_lap, 5, 9, 9)
    # cv2.imwrite("output1.png", dilate_lap)
    circles = cv2.HoughCircles(gray_lap, cv2.HOUGH_GRADIENT, 1, 1, param1=400, param2=22, minRadius=70, maxRadius=80)
    # print(circles)
    borderimg = draw_marker(gray, circles)
    print("{} circles detected.".format(circles[0].shape[0]))
    # There are some false positives left in the regions containing the numbers.
    #  We can experiment more with other filters and see what works best.
    return borderimg

borderimg = detect_circular_objects("output4.png")
cv2.imwrite("outputme.png", borderimg)
# cv2.imshow("images", borderimg)
# cv2.resizeWindow('images', 600,600)
# cv2.waitKey(0)