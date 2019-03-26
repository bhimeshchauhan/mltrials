# import the necessary packages
import numpy as np
from skimage import color
from colour import Color
import cv2
import webcolors
import math
from PIL import Image, ImageCms

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    print("------>>>>", _)
    print("------????", hist)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0
    col = []

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, colour) in zip(hist, centroids):
        colorList = list(colour.astype("uint8"))
        colorLab = color.rgb2lab([[colorList]])

        # plot the relative percentage of each cluster
        isGreen = classify(colorLab.tolist()[0][0][1])

        if(isGreen):
            print("color #####", tuple(colour.astype("uint8")), isGreen)
            print("per #####", percent * 100)
        #     print("color ceil #####", color.bgr2lab(list(color.astype("uint8"))))
        #     # print("lol #####", webcolors.rgb_to_name(tuple(color.astype("uint8"))))
        #     print("000000000000000")
            col.append(percent)
        # print( "nope ",   webcolors.rgb_to_name(list(color.astype("uint8"))),percent*100)
        # endX = startX + (percent * 300)
        # cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
        # startX = endX

    # return the bar chart
    # return bar
    return col

def classify(aValue):
    # print("checking for", aValue)
    if(aValue < -1):
        return True

