import pydicom as dicom
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import stats
import scipy.ndimage
import imutils
import glob
from PIL import Image

def read_dicom(path):

    slices = [dicom.read_file(file,force=True).pixel_array for file in sorted(glob.glob(path + '*.dcm'))]

    return slices

def convert_grayscale(slices):

    slices_gray = []

    for i in range(len(slices)):
        img = slices[i].astype(float)
        img_2d = (np.maximum(img,0) / img.max()) * 255
        img_2d = np.uint8(img_2d)
        slices_gray.append(img_2d)

    return slices_gray

def align_images(dry_slices,wet_slices,maxFeatures=500,keepPercent= 0.2, debug=False):
    
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
        if (debug == True and s == 1000):
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

def center_image(slices):

    centered_slices = []

    for s in range(len(slices)):
        
        height, width = slices[s].shape
        wi, he = width/2, height/2
        ret,thresh = cv2.threshold(slices[s], 110, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        M = cv2.moments(thresh)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        dx = wi-cx
        dy = he-cy

        T = np.float32([[1,0,dx],[0,1,dy]])
        dst = cv2.warpAffine(slices[s],T,(width,height))

        centered_slices.append(dst)
    
    return centered_slices

def detect_circles(image):
    x, y, r = 0, 0, 0
    output = image.copy()
    output = cv2.medianBlur(output, 5)
    output = cv2.threshold(output, 110, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    circles = cv2.HoughCircles(image, method = cv2.HOUGH_GRADIENT, dp = 5, minDist = 500,maxRadius=70,minRadius=65)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("uint8")
        for (x,y,r) in circles:
            print(x,y,r)
            
    return x,y,r

def circles_list(slices):
    x, y, r = [], [], []
    for s in range(len(slices)):
        output = slices[s].copy()
        output = cv2.medianBlur(output, 5)
        output = cv2.threshold(output, 110, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        circles = cv2.HoughCircles(slices[s], method = cv2.HOUGH_GRADIENT, dp = 5, minDist = 200,maxRadius=70,minRadius=65)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("uint8")
            for (x1,y1,r1) in circles:
                x.append(x1)
                y.append(y1)
                r.append(r1)

    x_mode = stats.mode(x)
    y_mode = stats.mode(y)
    r_mode = stats.mode(r)

    return x_mode[0][0], y_mode[0][0], r_mode[0][0]

def mask_images(slices):
    masked_slices = []
    x_mode, y_mode, r_mode = circles_list(slices)

    for s in range(len(slices)):
        x,y,r = detect_circles(slices[s])

        if x != 252:
            x = 252
        if y != 248:
            y = 248

        # create a mask
        mask = np.full((slices[s].shape[0], slices[s].shape[1]), 0, dtype=np.uint8) 
        # create circle mask, center, radius, fill color, size of the border
        cv2.circle(mask,(x,y), r, (255,255,255),-1)
        # get only the inside pixels
        fg = cv2.bitwise_or(slices[s], slices[s], mask=mask)
        
        mask = cv2.bitwise_not(mask)
        background = np.full(slices[s].shape, 255, dtype=np.uint8)
        bk = cv2.bitwise_or(background, background, mask=mask)
        final = cv2.bitwise_or(fg, bk)
        cropped = final[150:350,150:350]
        masked_slices.append(cropped)
        
    return masked_slices

def resample(slices, new_spacing=[1,1,1]):
    resampled = []
    spacing = np.array([0.25,0.25,0.25])
    slices = np.dstack(slices)
    resize_factor = np.divide(spacing,new_spacing)
    new_real_shape = np.multiply(slices.shape, resize_factor)
    real_resize_factor = np.divide(new_real_shape,slices.shape)
    new_spacing = np.divide(spacing,real_resize_factor)


    resampled = scipy.ndimage.interpolation.zoom(slices, real_resize_factor)
        
    return resampled

def z_profiling(slices):
    z_profile = []
    number = []
    for s in range(len(slices)):
        z_profile.append(np.mean(slices[s]))
        number.append(s)
    
    plt.plot(number,z_profile)
    plt.xlabel('Slice Number')
    plt.ylabel('Mean CT')
    plt.title('Z-Profile')
    plt.ylim(0,255)
    plt.show()

    return z_profile
def save_tif(slices,path):
    for s in range(len(slices)):
        im = Image.fromarray(slices[s])
        im.save(path + str(s) + '.tif')

    return print('Done')

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

