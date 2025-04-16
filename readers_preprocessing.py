import pydicom as dicom
import numpy as np
import scipy.ndimage
import glob
from PIL import Image
import SimpleITK as sitk
import os 

def list_files_in_folder(folder_path, include_extension=False, include_hidden=False):
    """
    Returns a list of file names within the specified folder, with or without extensions.
    
    Parameters:
    folder_path (str): Path to the folder
    include_extension (bool): Whether to include file extensions in the returned names
                             (default: False)
    include_hidden (bool): Whether to include hidden files like ._* files (default: False)
    
    Returns:
    list: List of file names (not including directories)
    """
    try:
        # Check if the path exists and is a directory
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"The folder path '{folder_path}' does not exist")
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"'{folder_path}' is not a directory")
            
        # Get all items in the directory
        all_items = os.listdir(folder_path)
        
        # Filter to only include files (not directories)
        files_only = [item for item in all_items 
                     if os.path.isfile(os.path.join(folder_path, item))]
        
        # Filter out hidden files if requested
        if not include_hidden:
            files_only = [file for file in files_only if not file.startswith('._')]
        
        # Remove extensions if required
        if not include_extension:
            # Use a set to eliminate duplicates when extensions are removed
            files_only = list(set([os.path.splitext(file)[0] for file in files_only]))
        
        return files_only
    
    except Exception as e:
        print(f"Error: {e}")
        return []
    

def list_folders(directory_path, modified=False):
    """
    Returns a list of folder names within the specified directory.
    If modified=True, returns only the characters before the second underscore.
    
    Args:
        directory_path (str): Path to the directory to scan
        modified (bool): Whether to modify names to get text before second underscore
        
    Returns:
        list: List of folder names (not including files)
    """
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist.")
        return []
    
    # Check if the path is actually a directory
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a directory.")
        return []
    
    # Get all folders (excluding files)
    folders = [item for item in os.listdir(directory_path) 
               if os.path.isdir(os.path.join(directory_path, item))]
    
    # If modified flag is True, extract text before second underscore
    if modified:
        modified_folders = []
        for folder in folders:
            # Find positions of underscores
            first_underscore = folder.find('_')
            if first_underscore != -1:
                # Find second underscore only if first was found
                second_underscore = folder.find('_', first_underscore + 1)
                if second_underscore != -1:
                    # Get text up to second underscore
                    modified_folders.append(folder[:second_underscore])
                else:
                    # If there's no second underscore, keep full name
                    modified_folders.append(folder)
            else:
                # If there's no underscore at all, keep full name
                modified_folders.append(folder)
        return modified_folders
    
    return folders

def get_text_before_second_underscore(text):
    # Find the position of the first underscore
    first_underscore = text.find('_')
    
    # If first underscore exists, look for second underscore
    if first_underscore != -1:
        # Find second underscore, starting search after first underscore
        second_underscore = text.find('_', first_underscore + 1)
        
        # If second underscore exists, return everything before it
        if second_underscore != -1:
            return text[:second_underscore]
    
    # Return original string if there aren't two underscores
    return text


def read_dicom(path):
    '''
    Function that reads DICOM files and returns a numpy array of the DICOM files
    Inputs:
        path: path to the folder containing the DICOM files
    
    Outputs:
        slices: numpy array of the DICOM files, size = (NxNxM)
    '''
    slices = [dicom.dcmread(file,force=True).pixel_array for file in sorted(glob.glob(path + '*.dcm'))]
    slices = np.dstack(slices)
    slices = np.flip(slices,2)
    return slices

def save_slices_as_numpy(slices_dict, path, overwrite=False):
    '''
    Function that saves a dictionary of numpy arrays as numpy files without overwriting existing files
    
    Inputs:
        slices_dict: dictionary of numpy arrays of the slices, size = (NxNxM)
        path: path to the folder where the numpy files will be saved
        overwrite: boolean to indicate whether to overwrite existing files (default: False)
    
    Outputs:
        Numpy files saved in the specified folder
        Returns a list of files that were not saved due to existing files
    '''
    # Make sure path ends with a slash
    if not path.endswith('/') and not path.endswith('\\'):
        path = path + '/'
    
    # Check if the directory exists, if not create it
    if not os.path.exists(path):
        os.makedirs(path)
    
    skipped_files = []
    saved_files = []
    
    for key in slices_dict:
        file_path = path + key + '.npy'
        
        # Check if the file already exists
        if os.path.exists(file_path) and not overwrite:
            skipped_files.append(key)
            continue
        
        # Save the file
        np.save(file_path, slices_dict[key])
        saved_files.append(key)
    
    print(f"Saved {len(saved_files)} files.")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} existing files: {', '.join(skipped_files)}")
    
    return saved_files, skipped_files

def read_slices_from_numpy(path,files):
    '''
    Function that reads a dictionary of numpy arrays from numpy files
    Inputs:
        path: path to the folder where the numpy files are saved
    
    Outputs:
        slices_dict: dictionary of numpy arrays of the slices, size = (NxNxM)
    '''
    slices_dict = {}
    for file in files:
        slices_dict[file] = np.load(path + file + '.npy',allow_pickle=True)

    return slices_dict

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

def mask_images(slices,radius,first_slice,last_slice):
    '''
    This functions creates a circular mask around the center of the slices
    Inputs:
        slices: numpy array of the slices, size = (NxNxM)
        radius: radius of the circle
        first_slice: first slice to be included in the mask
        last_slice: last slice to be included in the mask

    Outputs:
        final: numpy array of the masked slices, size = (NxNxM)
    '''
    slices_output = slices.copy()
    height, width = slices.shape[0], slices.shape[1]
    cx = height/2 ; cy = width/2
    # create circular mask
    y, x = np.ogrid[:height, :width]
    distance_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = distance_from_center <= radius
    #circle_array = np.zeros((height,width))
    circle_array = np.full((height,width),np.nan)
    circle_array[mask] = 1

    # apply mask to slices
    for i in range(slices.shape[2]):
        slices_output[:,:,i] = slices[:,:,i]*circle_array

    final = slices_output[int(cy-radius):int(cy+radius),int(cx-radius):int(cx+radius),first_slice:last_slice]

    return final

def apply_gaussian(slices,radius=1,order=0,sigma=2):
    '''
    Function that applies a Gaussian filter to the slices
    Inputs:
        slices: numpy array of the slices, size = (NxNxM)
        radius: radius of the Gaussian filter
        order: order of the Gaussian filter (0 = Gaussian, 1 = first derivative, 2 = second derivative)
        sigma: standard deviation of the Gaussian filter

    Outputs:
        slices_output: numpy array of the filtered slices, size = (NxNxM)
    '''
    slices_output = slices.copy()

    slices_output = scipy.ndimage.gaussian_filter(slices_output,radius=radius,sigma=sigma,order=order)

    return slices_output

def resample(slices,size=3,resolution = [0.25,0.25,0.25], new_spacing=[1,1,1]):
    '''
    Voxel coarsening scheme to reduce uncertainty in the CT values. Median filter is
    used to denoise the saturation and reduce uncertainty. This function will smooth the slices and 
    resample them to a new spacing.
    Square voxel is suggested for the resampling.

    Inputs:
        slices: numpy array of the slices, size = (NxNxM)
        size: size of the median filter
        resolution: voxel dimensions of the slices
        new_spacing: new voxel dimensions

    Outputs:
        resampled: numpy array of the resampled slices, size = (NxNxM)
    '''
    slices = scipy.ndimage.median_filter(slices,size=size)
    resampled = []
    spacing = np.array(resolution)
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
    # Translation variables
    length = scans.shape[2]
    width = scans.shape[0]
    cx = width/2
    cy = width/2
    dx = (cxf-cxi)/length
    dy = (cyf-cyi)/length

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
        vector = ((center[i][0]-cx),(center[i][1]-cy)) # the sign defines the orientation of the translation (here, it is opposite) negative move right and down, positive move left and up
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





