import pydicom as dc
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imutils

# read dry and wet image
dry = dc.dcmread('C:/Users/ubillusj/Desktop/Almostafa/Experiment_2/Dry_slow_1/0009.dcm')
wet = dc.dcmread('C:/Users/ubillusj/Desktop/Almostafa/100%_brine/Raw_wet/0004.dcm')

# get pixel array
dry_array = dry.pixel_array
wet_array = wet.pixel_array

img_dry = dry_array.astype(float)
img_2d_dry = (np.maximum(img_dry,0) / img_dry.max()) * 255.0
img_2d_dry = np.uint8(img_2d_dry)

img = wet_array.astype(float)
img_2d = (np.maximum(img,0) / img.max()) * 255.0
img_2d_wet = np.uint8(img_2d)

def transform_to_hu(raw,array):

    intercept = raw.RescaleIntercept

    slope = raw.RescaleSlope

    hu_image = array * slope + intercept

    return hu_image
    
def align_images(dry_array, wet_array, maxFeatures=5000, keepPercent=0.5,debug=False):

    dry_array = cv2.bilateralFilter(dry_array,9,75,75)
    wet_array = cv2.bilateralFilter(wet_array,9,75,75)

    ## ORB detector (keypoints and descriptors)
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(dry_array, None)
    (kpsB, descsB) = orb.detectAndCompute(wet_array, None)

    ## Match features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #matcher = cv2.DescriptorMatcher_create(method,crosscheck=True)
    matches = matcher.match(descsA, descsB, None)

    ## Sort matches by score
    matches = sorted(matches, key=lambda x:x.distance,reverse = False)

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
    #(H, mask) = cv2.findHomography(ptsB, ptsA, method=cv2.RANSAC)
    (H, mask) = cv2.estimateAffinePartial2D(ptsB, ptsA, method=cv2.RANSAC)
    ## Use the homography matrix to align the images
    (h, w) = wet_array.shape[:2]
    
    #aligned = cv2.warpPerspective(wet_array, H, (w, h))
    aligned = cv2.warpAffine(wet_array, H, (w, h))

    ## Return the aligned image
    return aligned

#aligned = align_images(img_2d_dry, img_2d_wet, debug=True)

def center_image(image):
  crop_img = image[50:450, 50:450]
  height, width = crop_img.shape
  print(crop_img.shape)
  wi=(width/2)
  he=(height/2)
  print(wi,he)
  #ret,thresh = cv2.threshold(image,95,255,0)

  M = cv2.moments(crop_img)

  cX = int(M["m10"] / M["m00"])
  cY = int(M["m01"] / M["m00"])

  offsetX = (wi-cX)
  offsetY = (he-cY)
  T = np.float32([[1, 0, offsetX], [0, 1, offsetY]]) 
  centered_image = cv2.warpAffine(crop_img, T, (width, height))

  return centered_image

#aligned = center_image(aligned)

def detect_circles(image):
    fx,fy,fr = 0,0,0
    img_dry = image.astype(float)
    img_2d_dry = (np.maximum(img_dry,0) / img_dry.max()) * 255.0
    img_2d_dry = np.uint8(img_2d_dry)
    circles = cv2.HoughCircles(img_2d_dry, method = cv2.HOUGH_GRADIENT, dp = 5, minDist = 500,maxRadius=70,minRadius=65)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        fx, fy , fr = circles[0]
    
    return fx, fy, fr

def crop_image(image):

    x, y ,r = detect_circles(image)
    print(x,y,r)

    # create a mask
    mask = np.full(image.shape, 0, dtype=np.uint8) 
    # create circle mask, center, radius, fill color, size of the border
    cv2.circle(mask,(x,y), r, (255,255,255),-1)
    # get only the inside pixels
    fg = cv2.bitwise_or(image, image, mask=mask)
    
    mask = cv2.bitwise_not(mask)
    background = np.full(image.shape, 255, dtype=np.uint8)
    bk = cv2.bitwise_or(background, background, mask=mask)
    final = cv2.bitwise_or(fg, bk)

    return plt.imshow(final, cmap='gray')


#plt.imshow(aligned, cmap=plt.cm.bone)
test = center_image(dry_array)
crop_image(test)