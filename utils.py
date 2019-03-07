# import the necessary packages
import numpy as np
import cv2
import webcolors


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
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster

        green = classify(tuple(color.astype("uint8")))
        if(green is "green"):
            print("color #####", list(color.astype("uint8")), color.astype(np.float64))
            print("per #####", percent * 100)
            print("000000000000000")
            col.append(percent)
        # print( "nope ",   webcolors.rgb_to_name(list(color.astype("uint8"))),percent*100)
        # endX = startX + (percent * 300)
        # cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
        # startX = endX

    # return the bar chart
    # return bar
    return col

def classify(rgb_tuple):
    # eg. rgb_tuple = (2,44,300)

    # add as many colors as appropriate here, but for
    # the stated use case you just want to see if your
    # pixel is 'more red' or 'more green'
    colors = {"not green": (255, 0, 0),
              "green" : (0,255,0),
              }

    manhattan = lambda x,y : abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2])
    distances = {k: manhattan(v, rgb_tuple) for k, v in colors.items()}
    color = min(distances, key=distances.get)
    return color