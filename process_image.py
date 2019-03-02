#!/usr/bin/python3

import numpy as np
import cv2
import argparse
from os.path import basename
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed


# reading the arguments from the command line -i, or --image 
# app = argparse.ArgumentParser()
# app.add_argument("-fn", "--filename", required=True,
# 	help="Name of the file to be processed")
# app.add_argument("-o", "--output", required=True,
# 	help="Output file name")
# args = vars(app.parse_args())

########## variables
# OUTPUT_FILE_NAME = args["output"]
########## 


# default values for the helper function
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_color = (0, 0, 0)
margin = 5
thickness = 1.2
text_background = (255, 255, 255)
# helper function to put text on the image


def putTextWithBoundingBox(image, text, x, y, font, font_scale, color, thickness):

    text_size = cv2.getTextSize(text, font, font_scale, thickness)
    text_width = text_size[0][0]
    text_height = text_size[0][1]
    x_e = x + text_width
    y_e = y + text_height
    cv2.rectangle(image, (x - margin - int(text_width/2) ,y - margin), (x_e + margin - int(text_width/2), y_e + margin), text_background, thickness = -1)
    cv2.putText(image, text, (x - int(text_width/2), y + text_height), font, font_scale, color, thickness)


def normalize_image(raw_image):
    # r = R / R + G + B
    # g = G / R + G + B
    # b = B / R + G + B

    a = raw_image.sum(axis=2)
    #remove zeros
    a[a == 0] = 1 
    image_b = cv2.split(raw_image)[0] / a
    image_g = cv2.split(raw_image)[1] / a
    image_r = cv2.split(raw_image)[2] / a

    # scale it up to 0-255 range 
    norm_b = image_b * 255
    norm_g = image_g * 255
    norm_r = image_r * 255

    return cv2.merge((norm_b, norm_g, norm_r)).astype(np.uint8)

def find_markers(raw_image):
    ########## variables
    MINIMUN_CONTOUR_AREA_BLUE = 850
    points=[]
    ##########
    # color profile for Blue markers in HSV OpenCV: For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
    lower_blue_profile = np.array([100, 70, 105], dtype="uint8")
    upper_blue_profile = np.array([120, 255, 255], dtype="uint8")
    ##########
    # converting the color space to HSV
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)
    # raw_image = cv2.blur(raw_image, (5, 5))

    mask_blue = cv2.inRange(raw_image, lower_blue_profile, upper_blue_profile)
    # Creaing element, and dialate and erode to remove unwanted pixel
    kernel = np.ones((3,3), 'uint8')
    dilate = cv2.dilate(mask_blue, kernel, iterations=1)
    mask_blue = cv2.erode(dilate, kernel, iterations=1)
    # cv2.imwrite("find_markers.jpg", mask_blue)
    # find the markers 
    contours,_ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)        
        # cv2.drawContours(mask_yellow, [c], -1, (0,0,255), 2)
        if area < MINIMUN_CONTOUR_AREA_BLUE: # removing false positives 
            continue
        M = cv2.moments(c)

        if M['m00'] == 0:
            cx = 0
            cy = 0
        else:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            points.append([cx,cy])
    return points
    
def crop_image_based_on_markers(raw_image, points):

    # Finding the location of the points
    x_max, _ = max(points,key=lambda item:item[0])
    _, y_max = max(points,key=lambda item:item[1])
    x_min, _ = min(points,key=lambda item:item[0])
    _, y_min = min(points,key=lambda item:item[1])


    # dividing the plane in to four sections to find which point is where
    # to locate the points of the rectangle. x_center,y_center are x,y of center of the 4 points
    x_center = int((x_max - x_min) / 2 + x_min)
    y_center = int((y_max - y_min) / 2 + y_min)

    # finding location of the points
    point_nw = [0,0]
    point_ne = [0,0]
    point_sw = [0,0] 
    point_se = [0,0]

    for p in points:
        x , y = p
        if (x < x_center): # means that we are on the west side
            if (y < y_center): # means north west
                point_nw = [x,y]
            else:
                point_sw = [x,y]
        else: # on the east side
            if (y < y_center): # means north east
                point_ne = [x,y]
            else:
                point_se = [x,y]

    # arranging point to calculate the transformation matrix
    pts1 = np.float32([point_nw, point_ne, point_sw, point_se])
    pts2 = np.float32([[0,0],[x_center - x_min,0],[0,y_center - y_min],[x_center - x_min,y_center - y_min]])

    # finding the transformation matrix
    M = cv2.getPerspectiveTransform(pts1,pts2)
    # cv2.imwrite('cropped.jpg', cv2.warpPerspective(raw_image, M, (x_center - x_min, y_center - y_min)))
    # geometric transformation based on the transformation matrix
    return cv2.warpPerspective(raw_image, M, (x_center - x_min, y_center - y_min))

def pixels_of_plants_in_image(name, cropped_image):
    # ########## variables
    # ## in OpenCV b ranges from -127 to 127 (in L*a*b* color space). So, (b + 128) is used for thresholding
    # RED_THRESHOLD = 29+128
    # GREEN_THRESHOLD = 154
    # MINIMUN_CONTOUR_AREA = 3150
    # LOCAL_MAXIMA_MIN_DISTANCE = 55
    markers_distance = 46 #inches
    # index = -1
    # thickness = 2
    # color = (0,0,255)
    # total_area = 0
    # ##########
    #
    # # applying mean shift the uniform the colors for better segmentation and get rid fine textures
    # shifted = cv2.pyrMeanShiftFiltering(cropped_image, 21, 31)
    #
    # # change the color space to L*a*b
    # imglab = cv2.cvtColor(shifted, cv2.COLOR_BGR2Lab)
    #
    # # separate image channels
    # l,a,b = cv2.split(imglab)
    #
    # lower_profile = np.array([120], dtype="uint8")
    # upper_profile = np.array([165], dtype="uint8")
    # '''
    # # thresholding channel b of the image
    # _, mask_green = cv2.threshold(b, GREEN_THRESHOLD, 255, cv2.THRESH_BINARY)
    #
    # # this part for red/purplish leaves (130 is good) ##TODO: find threshold based on the image itself
    # _, mask_red = cv2.threshold(b, RED_THRESHOLD, 255, cv2.THRESH_BINARY)
    #
    # # combining two masks so we include green and red leaves
    # mask_combined = cv2.bitwise_or(mask_green, mask_red)
    # '''
    # mask_combined = cv2.inRange(b, lower_profile, upper_profile)
    # # doing morphological closing to fill  up the holes in the connected components
    # kernel = np.ones((5,5),np.uint8)
    # mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
    # mask_combined = cv2.bitwise_not(mask_combined)
    # cv2.imwrite("mask.jpg", mask_combined)
    # # compute the exact Euclidean distance from every binary
    # # pixel to the nearest zero pixel, then find peaks in this
    # # distance map
    # D = ndimage.distance_transform_edt(mask_combined)
    # localMax = peak_local_max(D, indices=False, min_distance=LOCAL_MAXIMA_MIN_DISTANCE, labels=mask_combined)
    #
    # # perform a connected component analysis on the local peaks,
    # # using 8-connectivity, then appy the Watershed algorithm
    # markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    # labels = watershed(-D, markers, mask=mask_combined)
    #
    # # calculate the ratio of pixels to inches
    pixels = cropped_image.shape[1] # width of the corrected image
    ratio = markers_distance/pixels
    ratio = ratio * ratio # to make it area instead of line (2d)
    #
    # # loop over the unique labels returned by the Watershed algorithm
    # for label in np.unique(labels):
    #     # if the label is zero, we are examining the 'background' so simply ignore it
    #     if label == 0:
    #         continue
    #
    #     # otherwise, allocate memory for the label region and draw it on the mask
    #     mask = np.zeros(mask_combined.shape, dtype="uint8")
    #     mask[labels == label] = 255
    #
    #     # detect contours in the mask and grab the largest one
    #     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    #     c = max(cnts, key=cv2.contourArea)
    #     area = cv2.contourArea(c)
    #     if area < MINIMUN_CONTOUR_AREA:	 # removing false positives
    #         continue
    #     # draw a circle enclosing the object
    #     ((cx, cy), _) = cv2.minEnclosingCircle(c)
    #     cv2.drawContours(cropped_image, [c], index, color, thickness)
    #     putTextWithBoundingBox(cropped_image,"#{}: {}".format(label, round(area * ratio, 2)), int(cx), int(cy), cv2.FONT_HERSHEY_SIMPLEX, .4, (0,0,0), 1)
    #     total_area = total_area + (area * ratio)
    # write_image_to_file(cropped_image)
    # return round(total_area,2)

    # blur = cv2.GaussianBlur(cropped_image, (15, 15), 2)
    blur = cv2.bilateralFilter(cropped_image, 9 ,250,250)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV_FULL)
    lower_green = np.array([37, 0, 0])
    upper_green = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    masked_img = cv2.bitwise_and(cropped_image, cropped_image, mask=opened_mask)
    img_hsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV_FULL)

    # Filter out low saturation values, which means gray-scale pixels(majorly in background)
    bgd_mask = cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([255, 30, 255]))

    # Get a mask for pitch black pixel values
    black_pixels_mask = cv2.inRange(masked_img, np.array([0, 0, 0]), np.array([70, 70, 70]))

    # Get the mask for extreme white pixels.
    white_pixels_mask = cv2.inRange(masked_img, np.array([230, 230, 230]), np.array([255, 255, 255]))

    final_mask = cv2.max(bgd_mask, black_pixels_mask)
    final_mask = cv2.min(final_mask, ~white_pixels_mask)
    final_mask = ~final_mask

    final_mask = cv2.erode(final_mask, np.ones((3, 3), dtype=np.uint8))
    final_mask = cv2.dilate(final_mask, np.ones((5, 5), dtype=np.uint8))

    write_image_to_file(name+"_final.jpg", final_mask)
    # cv2.imwrite('final_mask.png', final_mask)
    # Now you can finally find contours.
    contours, hierarchy = cv2.findContours(final_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    final_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 1200 < area:
            final_contours.append(contour)

    for i in range(len(final_contours)):
        area = cv2.contourArea(final_contours[i])
        # compute the center of the contour
        M = cv2.moments(final_contours[i])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        src = cv2.drawContours(cropped_image, final_contours, i, np.array([50, 250, 50]), 4)
        cv2.putText(src, str(area), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # write_image_to_file('./final/', src)

def write_image_to_file(name, img):

    # write the input image matrix to file 
    # cv2.imwrite(OUTPUT_FILE_NAME, img)

    cv2.imwrite(name, img)
    return

def main():

    # read image from the input address 
    # img = cv2.imread(args["filename"])

    # check if the input file is readable
    # if img is None:
    #     print("[FATAL] The input file cannot be read.")
    #     exit()
    images = ["img/test1.jpg","img/test2.jpg", "img/test3.jpg", "img/test4.jpg", "img/test5.jpg", "img/test6.jpg", "img/test7.jpg",
              "img/test8.jpg","img/test9.jpg"]
    for imgname in images:
        img = cv2.imread(imgname)
        markers = find_markers(img)
        # Check if we have detected enough points for correcting the image. We need exactly 4 points
        # if len(markers) !=4 :
        #     print("[FATAL] {} markers found! We need exactly 4 of them!".format(len(markers)))
        #     continue
        # undistorted_and_cropped_image = crop_image_based_on_markers(img, markers)
        name = imgname.split("/")[1]
        strname = name.split(".")[0]
        pixels_of_plants_in_image(strname, img)
        # area = pixels_of_plants_in_image(undistorted_and_cropped_image)

        # return the area for the output
        # print(area)


# we do this to make sure that if this file is run from other files, it will behave as expected
if __name__ == "__main__":
    main()
