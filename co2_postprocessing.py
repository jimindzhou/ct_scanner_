import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

'''
This script contains functions specifically for estimation of porosity and  CO2 saturation in
core samples. The functions are divided into two categories: 1) functions for porosity estimation
and 2) functions for CO2 saturation estimation at the core level, slice level, and voxel level.

'''

def core_level_porosity(dry, wet):
    '''
    Function that estimates porosity at the core level assuming dry samples are fully saturated with air and
    wet samples are fully saturated with water.
    Inputs:
        dry: numpy array of the dry CT scans, size = (NxNxM)
        wet: numpy array of the wet CT scans, size = (NxNxM)
    Outputs:
        por_core: porosity at the core level
    
    '''
    ct_dry_avg = np.nanmean(dry)
    ct_wet_avg = np.nanmean(wet)

    por_core = (ct_wet_avg - ct_dry_avg) / (1000-0) # 1000 and 0 are the CT values of water and air, respectively

    print('Core level porosity estimate: ' + str(round(por_core,3)))

    return por_core

def slice_level_porosity(dry, wet):
    '''
    Function that estimates porosity at the slice level assuming dry samples are fully saturated with air and
    wet samples are fully saturated with water.
    Inputs:
        dry: numpy array of the dry CT scans, size = (NxNxM)
        wet: numpy array of the wet CT scans, size = (NxNxM)
    Outputs:
        por_slice: porosity at the slice level
    
    '''
    slice_dry_avg = np.nanmean(dry, axis=(0,1))
    slice_wet_avg = np.nanmean(wet, axis=(0,1))

    por_slice = (slice_wet_avg - slice_dry_avg) / (1000-0) # 1000 and 0 are the CT values of water and air, respectively

    fig, ax = plt.subplots()
    ax.plot(por_slice,range(len(por_slice)),color='k',linestyle="--",linewidth=2)
    ax.set_xlabel('Porosity')
    ax.set_ylabel('Slice')
    ax.set_xlim(0,0.5)
    ax.set_title('Slice-level porosity')

    plt.show()

    return por_slice

def voxel_level_porosity(dry, wet):
    '''
    Function that estimates porosity at the voxel level assuming dry samples are fully saturated with air and
    wet samples are fully saturated with water.
    Inputs:
        dry: numpy array of the dry CT scans, size = (NxNxM)
        wet: numpy array of the wet CT scans, size = (NxNxM)
    Outputs:
        por_voxel: porosity at the voxel level
    
    '''
    por_voxel = (wet - dry) / (1000-0) # 1000 and 0 are the CT values of water and air, respectively

    dims = por_voxel.shape
    s = np.linspace(0,dims[2]-1,9).astype(int)
    fig, axs = plt.subplots(3,3, figsize=(10, 10))
    for i in range(3):
        for j in range(3):
            im = axs[i, j].imshow(por_voxel[:, :, s[i * 3 + j]], cmap="gray")
            axs[i,j].set_title('Slice ' + str(s[i * 3 + j]))
            axs[i, j].axis("off")

    fig.colorbar(im, ax=axs.ravel().tolist(), orientation = 'vertical',shrink=0.5,pad=0.04)

    plt.show()
    return por_voxel

# Porosity and saturation estimations

def dashboard_porosity(por_core,por_slice,por_voxel):
    '''
    Function that builds a dashboard of porosity at all levels: core, slice, and voxel
    Inputs:
        por_core: porosity at the core level
        por_slice: porosity at the slice level
        por_voxel: porosity at the voxel level
    Outputs:
        Plots of the porosity estimations
    
    '''
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

def core_level_saturation(wet,co2,twophase):
    '''
    Function that estimates saturation at the core level assuming wet samples are fully saturated with brine,
    C02 samples are fully saturated with CO2 and twophase saturated with both fluids at multiple times.
    Inputs:
        wet: numpy array of the wet CT scans, size = (NxNxM)
        co2: numpy array of the CO2 CT scans, size = (NxNxM)
        twophase: dictionary of two-phase CT scans, size = (NxNxM)
    Output:
        sat_core: numpy array of saturation at multiple times, size = (1,length of dict)
    '''
    ctwater = np.nanmean(wet)
    ctco2 = np.nanmean(co2)
    sat_core = {}
    for key in twophase.keys():
        ctexpr = np.nanmean(twophase[key])
        sat_core[key] = (ctwater - ctexpr) / (ctwater - ctco2)
        

    fig,ax = plt.subplots(figsize = (5,5),dpi=200)
    ax.bar(np.arange(len(sat_core)),sat_core.values())
    ax.set_xlabel('Scan')
    ax.set_ylabel('Saturation')
    ax.set_title('Core level saturation')
    ax.set_xticks(range(len(sat_core)))
    ax.set_xticklabels(sat_core.keys())
    ax.set_ylim(0,1)
    plt.show()

    return sat_core
        
def compute_saturation_along_depth(wet,co2,twophase):
    '''
    Function that computes the saturation along the depth of the core.

    Inputs:
    wet: numpy array of the wet CT scans, size = (NxNxM)
    co2: numpy array of the CO2 CT scans, size = (NxNxM)
    twophase: dictionary of two-phase CT scans, size = (NxNxM)
    Outputs:
        sat_slice: dictionary of the saturation values along the axis, size = (1,M)

    '''
    
    ctco2 = np.nanmean(co2,axis=(0,1))
    ctwater = np.nanmean(wet,axis=(0,1))
    sat = {}
    for key in twophase.keys():
        sat[key] = (ctwater - np.nanmean(twophase[key],axis=(0,1)))/(ctwater - ctco2)
        #sat[key] = np.nanmean(sat[key],axis=(0,1))
    # Slice level saturation
 
    fig,ax = plt.subplots(figsize=(4,7))
    keys = list(sat.keys())
    length = len(sat[keys[0]])
        
    ax.fill_betweenx(np.arange(length),sat[keys[0]],label=keys[0])
    for j in range(0,len(keys)-1):
        ax.fill_betweenx(np.arange(length),sat[keys[j]],sat[keys[j+1]],label=keys[j+1])

    ax.set_title('Saturation along depth')
    ax.set_xlabel('Saturation')
    ax.set_xlim(0,1)
    ax.set_ylabel('Slice number')
    ax.legend(loc='center right',bbox_to_anchor=(1.5,0.5))
    ax.margins(0,0)
    plt.show()

    return sat

def voxel_level_saturation(wet,co2,twophase):
    '''
    Function that estimates saturation at the voxel level assuming wet samples are fully saturated with brine,
    C02 samples are fully saturated with CO2 and twophase saturated with both fluids at multiple times.
    Inputs:
        wet: numpy array of the wet CT scans, size = (NxNxM)
        co2: numpy array of the CO2 CT scans, size = (NxNxM)
        twophase: dictionary of two-phase CT scans, size = (NxNxM)
    Output:
        sat_voxel: numpy array of saturation at multiple times, size = (NxNxM,length of dict)
    '''

    ctco2 = np.nanmean(co2)
    ctwater = np.nanmean(wet)
    sat_voxel = {}
    for key in twophase.keys():
        sat_voxel[key] = (wet - twophase[key])/(ctwater - ctco2)
    
    return sat_voxel

def dashboard_saturation(wet,co2,twophase):
    '''
    Function that estimates saturation under three schemes: core level, slice level, and voxel level
    Inputs:
        wet: numpy array of the wet CT scans, size = (NxNxM)
        co2: numpy array of the CO2 CT scans, size = (NxNxM)
        twophase: numpy array of the two-phase CT scans, size = (NxNxM)
    Outputs:
        Plots of the saturation estimations
    
    '''
    # Core level saturation 
    ctexpr = np.nanmean(twophase)
    ctco2 = np.nanmean(co2)
    ctwater = np.nanmean(wet)

    sat_core = (ctwater - ctexpr) / (ctwater - ctco2)

    # Slice level saturation
    slice_expr_avg = np.nanmean(twophase, axis=(0,1))
    slice_co2_avg = np.nanmean(co2, axis=(0,1))
    slice_water_avg = np.nanmean(wet, axis=(0,1))

    sat_slice = (slice_water_avg - slice_expr_avg) / (slice_water_avg - slice_co2_avg)

    # Voxel-level saturation
    sat_voxel = (wet - twophase) / (ctwater - ctco2)

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
            im = axs[i, j+1].imshow(sat_voxel[:, :, s[i * 3 + j]], cmap="jet",interpolation='bilinear',vmin=0,vmax=1)
            axs[i,j+1].set_title('Slice ' + str(s[i * 3 + j]))
            axs[i, j+1].axis("off")
    
    fig.colorbar(im, ax=axs[0:,1:], orientation = 'vertical',shrink=0.5,pad=0.04,ticks=[0,0.2,0.4,0.6,0.8,1])
    fig.text(0.35,0.06,'Core level saturation estimate: ' + str(round(sat_core,3)),
             fontsize=14, bbox={'facecolor': 'lightblue', 'alpha': 0.5, 'pad': 5})

    return plt.show()