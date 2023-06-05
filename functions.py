import pydicom as dicom
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats 
import scipy.ndimage
import glob
from PIL import Image
import SimpleITK as sitk
import os

## Read file and normalizing functions

def read_dicom(path):
    'Returns a numpy array of the DCM files, size = (NxNxM)'
    slices = [dicom.read_file(file,force=True).pixel_array for file in sorted(glob.glob(path + '*.dcm'))]
    slices = np.dstack(slices)

    return slices

def read_tif(folder_path,total_scans):
    'Returns a numpy array of the TIF files, size = (NxNxM)'
    slices = []
    for i in range(total_scans):
        slices.append(np.array(Image.open(folder_path + str(i) + '.tif')))
    
    slices = np.dstack(slices)

    return slices

def mask_images(slices,cx,cy,radius,last_slice):
    slices_output = slices.copy()
    height, width = slices.shape[0], slices.shape[1]
    # create circular mask
    y, x = np.ogrid[:height, :width]
    distance_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = distance_from_center <= radius
    circle_array = np.zeros((slices.shape[0],slices.shape[1]))
    circle_array[mask] = 1

    # apply mask to slices
    for i in range(slices.shape[2]):
        slices_output[:,:,i] = slices[:,:,i]*circle_array

    final = slices_output[int(cy-radius):int(cy+radius),int(cx-radius):int(cx+radius),0:last_slice]

    return final

def resample(slices, new_spacing=[1,1,0.25]):
    resampled = []
    spacing = np.array([0.25,0.25,0.25])
    resize_factor = np.divide(spacing,new_spacing)
    new_real_shape = np.multiply(slices.shape, resize_factor)
    real_resize_factor = np.divide(new_real_shape,slices.shape)
    new_spacing = np.divide(spacing,real_resize_factor)

    resampled = scipy.ndimage.interpolation.zoom(slices, real_resize_factor,order=1)

    return resampled

def z_profiling(slices):
    z_profile = []
    number = []
    for s in range(slices.shape[2]):
        z_profile.append(np.mean(slices[:,:,s]))
        number.append(s)
    
    plt.plot(number,z_profile)
    plt.xlabel('Slice Number')
    plt.ylabel('Mean CT')
    plt.title('Z-Profile')
    plt.show()

    return z_profile

def histograms(slices):
    slices = slices[slices != 0 ]
    mu, std = stats.norm.fit(slices.flatten())

    # Plot the histogram.
    plt.hist(slices.flatten(),density=True,bins=100, color='c')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Histogram - Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.xlabel('CT')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()

    return 

def save_tif(slices,path):
    for s in range(slices.shape[2]):
        im = Image.fromarray(slices[:,:,s])
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

#### Simple ITK functions (hope this works better, trial 1)

def center_itk(scans,cxi,cyi,cxf,cyf):
    
    dx = (cxf-cxi)/scans.shape[2]
    dy = (cyf-cyi)/scans.shape[2]
    # Convert the scans into np.float32
    scans = np.float32(scans)

    # Convert the scans into SimpleITK format
    sitk_scans = sitk.GetImageFromArray(np.transpose(scans, (2, 0, 1)))

    # Get the center of each slice using the displacement
    center = []
    for i in range(sitk_scans.GetSize()[2]):
        center.append((round(cxi+dx*i),round(cyi+dy*i))) # This is the displacement of the center of the circle (reference is the first scan) (x,y)

    # Define the translation
    transform = sitk.TranslationTransform(2)

    # Apply translation transform to each slice to align them
    aligned_scans = []
    for i in range(sitk_scans.GetSize()[2]):
        vector = (-(256-center[i][0]),(256-center[i][1])) # the sign defines the orientation of the translation (here, it is opposite) negative move right and down, positive move left and up
        transform.SetParameters(vector)
        aligned_scan = sitk.Resample(sitk_scans[:,:,i], sitk_scans[:,:,i], transform, sitk.sitkLinear, 0.0)
        aligned_scans.append(sitk.GetArrayFromImage(aligned_scan))

    # Convert the list of aligned circles to a NumPy array
    aligned_array = np.array(np.transpose(aligned_scans, (1, 2, 0)))

    return center, aligned_array

def volume_registration(images_fixed,images_moving):

    # Convert the scans into np.float32
    images_fixed = np.float32(images_fixed)
    images_moving = np.float32(images_moving)

    # Convert the scans into SimpleITK format
    sitk_fixed = sitk.GetImageFromArray(np.transpose(images_fixed, (2, 0, 1)))
    sitk_moving = sitk.GetImageFromArray(np.transpose(images_moving, (2, 0, 1)))

    # Initialize CenteredTransformInitializer
    initial_transform = sitk.CenteredTransformInitializer(sitk_fixed, 
                                                          sitk_moving, 
                                                          sitk.Euler3DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.MOMENTS)
    
    # Define the registration method
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInterpolator(sitk.sitkLinear)

    # Define the initial transformation
    reg.SetInitialTransform(initial_transform, inPlace=False)

    # Run the registration
    final_transform = reg.Execute(sitk_fixed, sitk_moving)

    # Apply the transformation to the moving image
    moving_resampled = sitk.Resample(sitk_moving, sitk_fixed, final_transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)

    # Convert the registered image to a NumPy array
    registered_array = np.array(np.transpose(sitk.GetArrayFromImage(moving_resampled), (1, 2, 0)))

    return registered_array





