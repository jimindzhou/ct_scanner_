import pydicom as dicom
import numpy as np
import scipy.ndimage
import glob
from PIL import Image
import SimpleITK as sitk

def read_dicom(path):
    '''
    Function that reads DICOM files and returns a numpy array of the DICOM files
    Inputs:
        path: path to the folder containing the DICOM files
    
    Outputs:
        slices: numpy array of the DICOM files, size = (NxNxM)
    '''
    slices = [dicom.read_file(file,force=True).pixel_array for file in sorted(glob.glob(path + '*.dcm'))]
    slices = np.dstack(slices)
    slices = np.flip(slices,2)
    return slices

def read_tif(folder_path):
    '''
    Function that reads TIF files and returns a numpy array of the TIF files
    Inputs:
        folder_path: path to the folder containing the TIF files
    
    Outputs:
        slices: numpy array of the TIF files, size = (NxNxM)'''
    slices = []
    slices = [np.array(Image.open(folder_path + str(file) + '.tif')) for file in sorted(glob.glob(folder_path + '*.tif'))]
    slices = np.dstack(slices)
    slices = np.flip(slices,2)
    
    return slices

def mask_images(slices,cx,cy,radius,first_slice,last_slice):
    '''
    This functions creates a circular mask around the center of the slices
    Inputs:
        slices: numpy array of the slices, size = (NxNxM)
        cx: x coordinate of the center of the circle
        cy: y coordinate of the center of the circle
        radius: radius of the circle
        first_slice: first slice to be included in the mask
        last_slice: last slice to be included in the mask

    Outputs:
        final: numpy array of the masked slices, size = (NxNxM)
    '''
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

    final = slices_output[int(cy-radius):int(cy+radius),int(cx-radius):int(cx+radius),first_slice:last_slice]

    return final

def resample(slices,size = 3, new_spacing=[1,1,1]):
    '''
    Voxel coarsening scheme to reduce uncertainty in the CT values. This function
    will smooth the slices and resample them to a new spacing.
    Square voxel is suggested for the resampling.

    Inputs:
        slices: numpy array of the slices, size = (NxNxM)
        size: size of the median filter
        new_spacing: new voxel dimensions

    Outputs:
        resampled: numpy array of the resampled slices, size = (NxNxM)
    '''
    slices = scipy.ndimage.median_filter(slices,size=size)
    resampled = []
    spacing = np.array([0.25,0.25,0.25])
    resize_factor = np.divide(spacing,new_spacing)
    new_real_shape = np.multiply(slices.shape, resize_factor)
    real_resize_factor = np.divide(new_real_shape,slices.shape)
    new_spacing = np.divide(spacing,real_resize_factor)

    resampled = scipy.ndimage.interpolation.zoom(slices, real_resize_factor,order=0)

    return resampled

def save_tif(slices,path):
    '''
    Function that saves a numpy array of slices as TIF files
    Inputs:
        slices: numpy array of the slices, size = (NxNxM)
        path: path to the folder where the TIF files will be saved

    Outputs:
        TIF files saved in the specified folder
    '''
    for s in range(slices.shape[2]):
        im = Image.fromarray(slices[:,:,s])
        im.save(path + str(s) + '.tif')

    return print('Done')

def center_itk(scans,cxi,cyi,cxf,cyf):
    '''
    In some cases, the center of every scan will not be perfectly aligned.
    This function will align the center of every scan to the center of the first scan.
    Currently the scanner is perfectly aligned but you can always use this function to confirm.
    Inputs:
        scans: numpy array of the scans, size = (NxNxM)
        cxi: x coordinate of the center of the first scan
        cyi: y coordinate of the center of the first scan
        cxf: x coordinate of the center of the last scan
        cyf: y coordinate of the center of the last scan

    Outputs:
        aligned_array: numpy array of the aligned scans, size = (NxNxM)
    '''
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
        vector = ((center[i][0]-256),(center[i][1]-256)) # the sign defines the orientation of the translation (here, it is opposite) negative move right and down, positive move left and up
        transform.SetOffset(vector)
        aligned_scan = sitk.Resample(sitk_scans[:,:,i], sitk_scans[:,:,i], transform, sitk.sitkLinear, 0.0)
        aligned_scans.append(sitk.GetArrayFromImage(aligned_scan))

    # Convert the list of aligned circles to a NumPy array
    aligned_array = np.array(np.transpose(aligned_scans, (1, 2, 0)))

    return aligned_array

def volume_registration(images_fixed,images_moving):
    '''
    Sometimes during the experiment, the samples might slightly move.
    This function will align the volume of the moving scan to the volume of the fixed scan.
    Inputs:
        images_fixed: numpy array of the fixed scan, size = (NxNxM)
        images_moving: numpy array of the moving scan, size = (NxNxM)

    Outputs:
        registered_array: numpy array of the registered scan, size = (NxNxM)
    '''

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





