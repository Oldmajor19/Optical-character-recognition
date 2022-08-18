# Import the required packages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2

# Construct the arguement parser and pass the arguement
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to input image")
ap.add_argument('-east', '--east', type=str, help='path to input East text detector')
ap.add_argument('-c', '--min-confidence', type=float, default=0.5, help='Minimum probability required to inspect a region')
ap.add_argument('-w', '--width', type=int, default=320, help='resized image width (should be multiple of 32)')
ap.add_argument('-e', '--height', type=int, default=320, help='resized image height (should be multiple of 32)')
args = vars(ap.parse_args())

# Load the input image and grab the image dimensions
#path = r'C:\Users\pc\Desktop\Machine_learning\OCR\sample1.jpg'
#image = cv2.imread(path)
image = cv2.imread(args["image"])

orig = image.copy()
(H, W) = image.shape[:2]

# set the new width and height and then determine the ratio in change
(newH, newW) = (320, 320)

rW = W/ float(newW)
rH = H/ float(newH)
# resize the image and grab the new image dimension
image = cv2.resize(image,(newH, newW))

(H, W) = image.shape[:2]

cv2.imshow('image', image)
cv2.waitKey(0)

layerNames = [
    'feature_fusion/Conv_7/Sigmoid', # Output sigmoid activation which gives us probabili
    'feature_fusion/concat_3' # The second layer represents the geometry of the image. we're going to use it derive the bounding box for the text
]

# Let's load the open cv EAST text detector
net = cv2.dnn.readNet('frozen_east_text_detection.pb')

# Construct a blob from the image

blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),(123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))

(numRows, numCols) = scores.shape[2:4] # Get the number of rows and columns in our score value
rects = [] # Stores the bounding box (x, y)-coordinates for text regions
confidences = [] # Stores the probability associated with each of the bounding boxes in rects


# loop over the number of rows
for y in range(0, numRows):
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]

    for x in range(0, numCols):
        # if our score does not have sufficient probability, ignore it
        if scoresData[x] < 0.5:
            continue

        # compute the offset factor as our resulting feature maps will
        # be 4x smaller than the input image
        (offsetX, offsetY) = (x * 4.0, y * 4.0)

        # extract the rotation angle for the prediction and then
        # compute the sin and cosine
        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)

        # use the geometry volume to derive the width and height of
        # the bounding box
        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]

        # compute both the starting and ending (x, y)-coordinates for
        # the text prediction bounding box
        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)

        # add the bounding box coordinates and probability score to
        # our respective lists
        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])


# apply non-maxima suppression to suppress weak, overlapping bounding boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)


# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
    # scale the bounding box coordinates based on the respective
    # ratios
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # draw the bounding box on the image
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
# show the output image
cv2.imshow("Text Detection", orig)
cv2.waitKey(0)
