import matplotlib.pyplot as plt
import numpy as np
from scipy import stats 

def z_profiling(sdict):
    '''
    This function plots the the CT values along the z-axis for each scan
    Inputs:
        sdict: dictionary of the scans, size = (NxNxM)
    Outputs:
        z_profiles: dictionary of the z-profiles, size = (Mx1)
    '''
    z_profiles= {}
    for key in sdict.keys():
        number = np.arange(0,sdict[key].shape[2],1)
        sdict[key][sdict[key] == 0] = np.nan
        z_profiles[key] = np.nanmean(sdict[key],axis=(0,1))
        
        plt.plot(number,z_profiles[key],label=key)

    plt.legend()
    plt.xlabel('Slice Number')
    plt.ylabel('Mean CT')
    plt.title('Z-Profile')
    plt.show()

    return z_profiles

def histogram(slices):
    '''
    This function plots the histogram of the CT values for a given scan
    Inputs:
        slices: numpy array of the slices, size = (NxNxM)
    Outputs:
        Histogram of the CT values
    '''

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

def compare_scans(sdict,i=256):
    '''
    This function compares the CT values of the same slice for three set of scans
    Inputs:
        s1: numpy array of the slices, size = (NxNxM)
        s2: numpy array of the slices, size = (NxNxM)
        s3: numpy array of the slices, size = (NxNxM)
        i: slice along x axis to be compared

    Outputs:
        Comparison of the CT values of the same slice for three set of scans
    '''
    length = len(sdict)
    fig, axs = plt.subplots(1, length, figsize=(15, 15))
    for j, key in enumerate(sdict.keys()):
        axs[j].imshow(sdict[key][i], cmap="gray")
        axs[j].set_title(key)
        axs[j].axis("off")
    
    return plt.show()

# Plot multiple slices of the different scans
def plot_slices(slices, nx, ny):
    '''
    Function that generates a figure with nx by ny subplots of the slices of the scan
    Inputs:
        slices: numpy array of the slices, size = (NxNxM)
        nx: number of columns
        ny: number of rows
    Outputs:
        Figure with nx by ny subplots of the slices of the scan

    '''
    dims = slices.shape
    s = np.linspace(0,dims[2]-1,nx*ny).astype(int)
    fig, axs = plt.subplots(ny, nx, figsize=(15, 15))
    for i in range(ny):
        for j in range(nx):
            im = axs[i, j].imshow(slices[:, :, s[i * nx + j]], cmap="gray")
            axs[i,j].set_title('Slice ' + str(s[i * nx + j]))
            axs[i, j].axis("off")

    fig.colorbar(im, ax=axs.ravel().tolist(), orientation = 'vertical',shrink=0.5,pad=0.04)
    plt.tight_layout()

    return plt.show()

# Plot multiple histograms in one figure of different scans
def plot_multiple_histograms(s1,s2):
    '''
    Function that plots two histograms of the CT values of two different scans
    Inputs:
        s1: numpy array of the slices, size = (NxNxM)
        s2: numpy array of the slices, size = (NxNxM)
    Outputs:
        Two histograms of the CT values of two different scans
    '''
    s1 = s1[s1 != 0 ]
    s2 = s2[s2 != 0 ]
    mu1, std1 = stats.norm.fit(s1.flatten())
    mu2, std2 = stats.norm.fit(s2.flatten())

    fig, ax = plt.subplots(figsize=(5,5))
    ax.hist(s1.flatten(),density=True,bins=100, color='c',label='Scan 1 ' + ' (mean = ' + str(round(mu1,2)) + ', std = ' + str(round(std1,2)) + ')')
    ax.hist(s2.flatten(),density=True,bins=100, color='m',label='Scan 2 '+ ' (mean = ' + str(round(mu2,2)) + ', std = ' + str(round(std2,2)) + ')')

    xmin1 = mu1-3*std1 ; xmax1 = mu1+3*std1
    xmin2 = mu2-3*std2 ; xmax2 = mu2+3*std2

    x1 = np.linspace(xmin1, xmax1, 10000) ; x2 = np.linspace(xmin2, xmax2, 10000)
    p1 = stats.norm.pdf(x1, mu1, std1) ; p2 = stats.norm.pdf(x2, mu2, std2)
    ax.plot(x1, p1, 'k', linewidth=1,linestyle='--')
    ax.plot(x2, p2, 'k', linewidth=1,linestyle='--')

    ax.set_xlabel('CT')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0,2000)
    plt.legend(loc='upper right')

    return plt.show()