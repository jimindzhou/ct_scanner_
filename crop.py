import pydicom as dc
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imutils

# read dry and wet image
dry = dc.dcmread('1210_dry.dcm')
wet = dc.dcmread('1210_wet.dcm')

# get pixel array
dry_array = dry.pixel_array
wet_array = wet.pixel_array

img_dry = dry_array.astype(float)
img_2d_dry = (np.maximum(img_dry,0) / img_dry.max()) * 255.0
img_2d_dry = np.uint8(img_2d_dry)

img = dry_array.astype(float)
img_2d = (np.maximum(img,0) / img.max()) * 255.0
img_2d_wet = np.uint8(img_2d)


def align_images(dry_array, wet_array, maxFeatures=500, keepPercent=0.2,debug=False):

    ## ORB detector (keypoints and descriptors)
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(dry_array, None)
    (kpsB, descsB) = orb.detectAndCompute(wet_array, None)

    ## Match features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    ## Sort matches by score
    matches = sorted(matches, key=lambda x:x.distance)

    ## Keep top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    # Visualize matches
    if debug == True:
        matchedVis = cv2.drawMatches(dry_array, kpsA, wet_array, kpsB, matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

    ## Allocate memory for the keypoints (x, y)-coordinates from the top matches
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    ## Loop over the top matches
    for (i, m) in enumerate(matches):
        ## Get the matching keypoints for each of the images
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt
    
    ## Compute the homography matrix between the two sets of points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    ## Use the homography matrix to align the images
    (h, w) = wet_array.shape[:2]
    aligned = cv2.warpPerspective(dry_array, H, (w, h))

    ## Return the aligned image
    return aligned

aligned = align_images(img_2d_dry, img_2d_wet, debug=False)

