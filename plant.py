import sys
import cv2 as cv
import numpy as np
def main(argv):
    
    default_file = 'cropped.jpg'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    #################################################################################
    #################################################################################
    
    gray = cv.cvtColor(src, cv.COLOR_BGR2RGB)
    cv.imwrite("plant_color.png", gray)

    # hsv = cv.cvtColor(gray, cv.COLOR_BGR2HSV)
    # roi = hsv[430:450, 20:170]
    # mu, sig = cv.meanStdDev(roi)
    # a = 9

    # blue_mask = cv.inRange(hsv, mu-a*sig, mu+a*sig)

    # cv.imwrite("plant_color.png", blue_mask)


    #################################################################################

    #################################################################################

    blur = cv.GaussianBlur(src, (15, 15), 2)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV_FULL)
    lower_green = np.array([37, 0, 0])
    upper_green = np.array([179, 255, 255])
    mask = cv.inRange(hsv, lower_green, upper_green)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    opened_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    masked_img = cv.bitwise_and(src, src, mask=opened_mask)
    # cv.imshow('', masked_img)
    # cv.waitKey()
    cv.imwrite('masked_img.png', masked_img)

    #################################################################################

    #################################################################################

    img_hsv = cv.cvtColor(masked_img, cv.COLOR_BGR2HSV_FULL)

    # Filter out low saturation values, which means gray-scale pixels(majorly in background)
    bgd_mask = cv.inRange(img_hsv, np.array([0, 0, 0]), np.array([255, 30, 255]))

    # Get a mask for pitch black pixel values
    black_pixels_mask = cv.inRange(masked_img, np.array([0, 0, 0]), np.array([70, 70, 70]))

    # Get the mask for extreme white pixels.
    white_pixels_mask = cv.inRange(masked_img, np.array([230, 230, 230]), np.array([255, 255, 255]))

    final_mask = cv.max(bgd_mask, black_pixels_mask)
    final_mask = cv.min(final_mask, ~white_pixels_mask)
    final_mask = ~final_mask

    final_mask = cv.erode(final_mask, np.ones((3, 3), dtype=np.uint8))
    final_mask = cv.dilate(final_mask, np.ones((5, 5), dtype=np.uint8))
    cv.imwrite('final_mask.png', final_mask)
    # Now you can finally find contours.
    contours, hierarchy = cv.findContours(final_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    final_contours = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 1000:
            final_contours.append(contour)

    for i in range(len(final_contours)):
        area = cv.contourArea(final_contours[i])
        # compute the center of the contour
        M = cv.moments(final_contours[i])
        print("M -> ", M, M["m10"],  M["m00"], M["m01"] )
        cX = int(M["m10"] / M["m00"])
        print("cX -> ", cX)
        cY = int(M["m01"] / M["m00"])
        print("cY -> ", cY)
        print(i, " ---> ", area)
        src = cv.drawContours(src, final_contours, i, np.array([50, 250, 50]), 4)
        src = cv.putText(src, str(area) ,  (cX - 20, cY - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


    debug_img = src
    # debug_img = cv.resize(debug_img, None, fx=0.3, fy=0.3)
    cv.imwrite("./plant.png", debug_img)


    #################################################################################

    #################################################################################

    
if __name__ == "__main__":
    main(sys.argv[1:])