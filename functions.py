import pydicom as dicom
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
    slices = np.flip(slices,2)
    return slices

def read_tif(folder_path,total_scans):
    'Returns a numpy array of the TIF files, size = (NxNxM)'
    slices = []
    for i in range(total_scans):
        slices.append(np.array(Image.open(folder_path + str(i) + '.tif')))
    
    slices = np.dstack(slices)

    return slices

def mask_images(slices,cx,cy,radius,first_slice,last_slice):
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

def piecewise_average(slices1,slices2):
    'According to Pini paper averaging over slices reduces the uncertainty of the measurement'
    n = 2
    average = np.sum(slices1,slices2)/n
    
    return average

def resample(slices,size = 3, new_spacing=[1,1,0.25]):
    slices = scipy.ndimage.median_filter(slices,size=size)
    resampled = []
    spacing = np.array([0.25,0.25,0.25])
    resize_factor = np.divide(spacing,new_spacing)
    new_real_shape = np.multiply(slices.shape, resize_factor)
    real_resize_factor = np.divide(new_real_shape,slices.shape)
    new_spacing = np.divide(spacing,real_resize_factor)

    resampled = scipy.ndimage.interpolation.zoom(slices, real_resize_factor,order=0)

    return resampled

def z_profiling(sdict):
    z_profiles= {}
    for key in sdict.keys():
        number = np.arange(0,sdict[key].shape[2],1)
        sdict[key] = sdict[key][sdict[key] != 0]
        z_profiles[key] = np.mean(sdict[key])
        
        plt.plot(number,z_profiles[key],label=key)

    plt.legend()
    plt.xlabel('Slice Number')
    plt.ylabel('Mean CT')
    plt.title('Z-Profile')
    plt.show()

    return z_profiles

def histograms(slices):
    slices = slices[slices != 0 ]
    mu, std = stats.norm.fit(slices.flatten())

    # Plot the histogram.
    plt.hist(slices.flatten(),density=True,bins=100, color='c')
    xmin, xmax = mu - 3*std, mu + 3*std
    x = np.linspace(xmin, xmax, 10000)
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
        vector = ((center[i][0]-256),(center[i][1]-256)) # the sign defines the orientation of the translation (here, it is opposite) negative move right and down, positive move left and up
        transform.SetOffset(vector)
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

# Plot multiple slices of the different scans
def plot_slices(slices, nx, ny):
    dims = slices.shape
    s = np.linspace(0,dims[2]-1,nx*ny).astype(int)
    fig, axs = plt.subplots(ny, nx, figsize=(15, 15))
    for i in range(ny):
        for j in range(nx):
            im = axs[i, j].imshow(slices[:, :, s[i * nx + j]], cmap="gray")
            axs[i,j].set_title('Slice ' + str(s[i * nx + j]))
            axs[i, j].axis("off")

    fig.colorbar(im, ax=axs.ravel().tolist(), orientation = 'vertical',shrink=0.5,pad=0.04)

    return plt.show()

# Plot multiple histograms in one figure of different scans
def plot_multiple_histograms(s1,s2,label1,label2):
    s1 = s1[s1 != 0 ]
    s2 = s2[s2 != 0 ]
    mu1, std1 = stats.norm.fit(s1.flatten())
    mu2, std2 = stats.norm.fit(s2.flatten())

    fig, ax = plt.subplots(figsize=(5,5))
    ax.hist(s1.flatten(),density=True,bins=100, color='c',label=label1 + ' (mean = ' + str(round(mu1,2)) + ', std = ' + str(round(std1,2)) + ')')
    ax.hist(s2.flatten(),density=True,bins=100, color='m',label=label2 + ' (mean = ' + str(round(mu2,2)) + ', std = ' + str(round(std2,2)) + ')')

    xmin1 = mu1-3*std1
    xmax1 = mu1+3*std1
    xmin2 = mu2-3*std2
    xmax2 = mu2+3*std2

    x1 = np.linspace(xmin1, xmax1, 10000)
    x2 = np.linspace(xmin2, xmax2, 10000)
    p1 = stats.norm.pdf(x1, mu1, std1)
    p2 = stats.norm.pdf(x2, mu2, std2)
    ax.plot(x1, p1, 'k', linewidth=1,linestyle='--')
    ax.plot(x2, p2, 'k', linewidth=1,linestyle='--')

    ax.set_xlabel('CT')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0,2000)
    plt.legend(loc='upper right')

    return plt.show()

# Porosity and saturation estimations
def porosity(dry, wet):
    wet_temp = wet.copy()
    dry_temp = dry.copy()
    wet_temp[wet_temp == 0 ] = np.nan
    dry_temp[dry_temp == 0 ] = np.nan

    ct_dry_avg = np.nanmean(dry_temp)
    ct_wet_avg = np.nanmean(wet_temp)

    # Core level porosity
    por_core = (ct_wet_avg - ct_dry_avg) / (1000-0)

    # Slice level porosity
    slice_dry_avg = np.nanmean(dry_temp, axis=(0,1))
    slice_wet_avg = np.nanmean(wet_temp, axis=(0,1))

    por_slice = (slice_wet_avg - slice_dry_avg) / (1000-0)

    # Voxel-level porosity
    por_voxel = (wet - dry) / (1000-0)

    # plot both slice level and multiple-voxel level porosity
    fig, axs = plt.subplots(3,4, figsize=(15, 10),dpi=300)
    gs = axs[0,0].get_gridspec()
    for ax in axs[0:,0]:
        ax.remove()

    axbig = fig.add_subplot(gs[0:,0])
    axbig.plot(por_slice,range(len(por_slice)),color='k',linestyle="--",linewidth=2)
    axbig.set_xlabel('Porosity')
    axbig.set_ylabel('Slice')
    axbig.set_xlim(0,0.5)
    axbig.set_title('Slice-level porosity')

    s = np.linspace(0,por_voxel.shape[2]-1,12).astype(int)
    for i in range(3):
        for j in range(3):
            im = axs[i,j+1].imshow(por_voxel[:, :, s[i * 3 + j]], cmap="gray",vmin=0,vmax=0.5)
            axs[i,j+1].set_title('Slice ' + str(s[i * 3 + j]))
            axs[i, j+1].axis("off")
            
    fig.colorbar(im, ax=axs[0:,1:], orientation = 'vertical',shrink=0.5,ticks=[0,0.2,0.4,0.6,0.8,1])
    fig.text(0.35,0.06,'Core level porosity estimate: ' + str(round(por_core,3)),
             fontsize=14, bbox={'facecolor': 'lightblue', 'alpha': 0.5, 'pad': 5})
   
    return plt.show()

def saturation(wet,co2,twophase):
    wet_temp = wet.copy()
    co2_temp = co2.copy()
    tph_temp = twophase.copy()
    wet_temp[wet_temp == 0 ] = np.nan
    co2_temp[co2_temp == 0 ] = np.nan
    tph_temp[tph_temp == 0 ] = np.nan

    # Core level saturation 
    ctexpr = np.nanmean(tph_temp)
    ctco2 = np.nanmean(co2_temp)
    ctwater = np.nanmean(wet_temp)

    sat_core = (ctwater - ctexpr) / (ctwater - ctco2)

    # Slice level saturation
    slice_expr_avg = np.nanmean(tph_temp, axis=(0,1))
    slice_co2_avg = np.nanmean(co2_temp, axis=(0,1))
    slice_water_avg = np.nanmean(wet_temp, axis=(0,1))

    sat_slice = (slice_water_avg - slice_expr_avg) / (slice_water_avg - slice_co2_avg)

    # Voxel-level saturation
    sat_voxel = (wet - twophase) / (wet - co2)

    # plot both slice level and multiple-voxel level saturation
    fig, axs = plt.subplots(3,4, figsize=(15, 10),dpi=300)
    gs = axs[0,0].get_gridspec()
    for ax in axs[0:,0]:
        ax.remove()

    axbig = fig.add_subplot(gs[0:,0])
    axbig.plot(sat_slice,range(len(sat_slice)),color='k',linestyle="--",linewidth=2)
    axbig.set_xlabel('Saturation')
    axbig.set_ylabel('Slice')
    axbig.set_xlim(0,1)
    axbig.set_title('Slice-level saturation')

    s = np.linspace(0,sat_voxel.shape[2]-1,12).astype(int)
    for i in range(3):
        for j in range(3):
            im = axs[i, j+1].imshow(sat_voxel[:, :, s[i * 3 + j]], cmap="jet",vmin=0,vmax=1)
            axs[i,j+1].set_title('Slice ' + str(s[i * 3 + j]))
            axs[i, j+1].axis("off")
    
    fig.colorbar(im, ax=axs[0:,1:], orientation = 'vertical',shrink=0.5,pad=0.04,ticks=[0,0.2,0.4,0.6,0.8,1])
    fig.text(0.35,0.06,'Core level saturation estimate: ' + str(round(sat_core,3)),
             fontsize=14, bbox={'facecolor': 'lightblue', 'alpha': 0.5, 'pad': 5})

    return plt.show()





