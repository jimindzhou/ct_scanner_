import pydicom as dicom
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imutils
import glob

dry_path = 'C:/Users/ubillusj/Desktop/Almostafa/N2_dry/'
wet_path = 'C:/Users/ubillusj/Desktop/Almostafa/100%_brine/'

def read_dicom(path):

    slices = [dicom.read_file(file,force=True).pixel_array for file in sorted(glob.glob(path + '*.dcm'))]

    return slices

def convert_grayscale(slices):

    slices_gray = []

    for i in range(len(slices)):
        img = slices[i].astype(float)
        img_2d = (np.maximum(img,0) / img.max()) * 255.0
        img_2d = np.uint8(img_2d)
        slices_gray.append(img_2d)

    return slices_gray

def align_images(dry_slices,wet_slices,maxFeatures=500,keepPercent= 0.5, debug=False):
    
    wet_aligned = []

    for s in range(len(dry_slices)):

        ## ORB detector (keypoints and descriptors)

        orb =cv2.ORB_create(maxFeatures)
        (kpsA, descsA) = orb.detectAndCompute(dry_slices[s], None)
        (kpsB, descsB) = orb.detectAndCompute(wet_slices[s], None)

        ## Match features

        method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        matcher = cv2.DescriptorMatcher_create(method)
        matches = matcher.match(descsA, descsB, None)

        ## Sort matches by score

        matches = sorted(matches, key=lambda x:x.distance,reverse=False)

        ## Keep top matches
        keep = int(len(matches) * keepPercent)
        matches = matches[:keep]

        # Visualize matches
        if (debug == True and i == 10):
            matchedVis = cv2.drawMatches(dry_slices[s], kpsA, wet_slices[s], kpsB, matches, None)
            matchedVis = imutils.resize(matchedVis, width=1000)
            cv2.imshow("Matched Keypoints", matchedVis)
            cv2.waitKey(0)
        
        ## Allocate memory for the keypoints (x, y)-coordinates from the top matches\

        ptsA = np.zeros((len(matches), 2), dtype="float")
        ptsB = np.zeros((len(matches), 2), dtype="float")

        ## Loop over the top matches

        for (i, m) in enumerate(matches):
            ## Get the matching keypoints for each of the images
            ptsA[i] = kpsA[m.queryIdx].pt
            ptsB[i] = kpsB[m.trainIdx].pt

        ## Compute the homography matrix between the two sets of points

        (H, mask) = cv2.estimateAffinePartial2D(ptsB, ptsA, method=cv2.RANSAC)

        ## Use the homography matrix to align the images

        (h, w) = wet_slices[s].shape[:2]
        aligned = cv2.warpAffine(wet_slices[s], H, (w, h))

        ## Store aligned image in a new tuple

        wet_aligned.append(aligned)

    return wet_aligned



def compare_images(dry_slices,wet_slices,wet_aligned,i=100):

    fig = plt.figure(figsize=(20,20))
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(dry_slices[i],cmap='gray')
    ax1.title.set_text('Dry')
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(wet_slices[i],cmap='gray')
    ax2.title.set_text('Wet')
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(wet_aligned[i],cmap='gray')
    ax3.title.set_text('Aligned')
    
    return plt.show()



dry_slices = read_dicom(dry_path)
wet_slices = read_dicom(wet_path)

dry_slices = convert_grayscale(dry_slices)
wet_slices = convert_grayscale(wet_slices)

wet_aligned = align_images(dry_slices,wet_slices,debug=False)

compare_images(dry_slices,wet_slices,wet_aligned,i=10)