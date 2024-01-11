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
    sdict_copy = sdict.copy()
    for key in sdict_copy.keys():
        number = np.arange(0,sdict_copy[key].shape[2],1)
        z_profiles[key] = np.nanmean(sdict_copy[key],axis=(0,1))
        
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
    sl = slices.copy()
    data = sl.flatten()
    data = data[~np.isnan(data)]
    mu, std = stats.norm.fit(data)

    # Plot the histogram.
    plt.hist(data,density=True,bins=100, color='c')
    xmin, xmax = mu - 3*std, mu + 3*std
    x = np.linspace(xmin, xmax, 10000)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Histogram - Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.xlim(-4000,4000)
    plt.xlabel('CT')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()

    return 

def compare_scans(sdict,i=256):
    '''
    This function compares the CT values of the same slice for three set of scans
    Inputs:
        sdict: dictionary of the scans, size = (NxNxM)
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
    fig, axs = plt.subplots(ny, nx, figsize=(10, 10))
    for i in range(ny):
        for j in range(nx):
            im = axs[i, j].imshow(slices[:, :, s[i * nx + j]], cmap="gray")
            axs[i,j].set_title('Slice ' + str(s[i * nx + j]))
            axs[i, j].axis("off")

    fig.colorbar(im, ax=axs.ravel().tolist(), orientation = 'vertical',shrink=0.5,pad=0.04)

    return plt.show()

# Plot multiple histograms in one figure of different scans
def plot_multiple_histograms(sdict):
    '''
    Function that plots two histograms of the CT values of two different scans
    Inputs:
        sdict: dictionary of the scans, size = (NxNxM)
    Outputs:
        Histograms of the CT values scans
    '''
    scopy = sdict.copy()
    for key in sdict.keys():
        data = scopy[key].flatten()
        data = data[~np.isnan(data)]

        mu, std = stats.norm.fit(data)
        plt.hist(data,density=True,bins=100,label=key)
        xmin, xmax = mu - 3*std, mu + 3*std
        x = np.linspace(xmin, xmax, 10000)
        p = stats.norm.pdf(x, mu, std)
        l = ' mu = %.2f,  std = %.2f' % (mu, std)
        plt.plot(x, p, 'k', linewidth=0.5,label=l)
        plt.xlim(0,4000)
        plt.xlabel('CT')
        plt.ylabel('Frequency')
        plt.legend(loc='right',bbox_to_anchor=(1.6, 0.5))
        plt.title('Scans histograms')

    return plt.show()