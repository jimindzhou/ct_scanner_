# The University of Texas at Austin
# Hildebrand Department of Petroleum and Geosystems Engineering
## Core-Tomography data pre- and post-processing functions and examples
## Collaborators:
- Jose Ubillus | MS Student
- Almostafa Alhadi | PhD Student
- Dr. David DiCarlo | PI

### Before running the scripts
We recommend Anaconda as Python/Package manager. If you know how to create environments or already have one defined, make sure you have the following libraries installed in your environment:
- Pandas
- Numpy
- Matplotlib
- Pydicom
- Scipy
- SimpleITK

If you are new to Anaconda, please follow the instructions in https://conda.io/projects/conda/en/latest/user-guide/index.html to download and install Conda. Then, in the Conda prompt run the following code:
`conda env create -f ctscanner.yml` , the YAML is available in the repository

### Description
The repository contains Python scripts with functions available for CT data file processing, image processing and tutorials on how to use the different scripts.
Function scripts:
- readers_preprocessing.py : contains dicom file reader to numpy array, create circular mask around cores, resampling scheme (voxel coarsening), etc.
- plot_scans.py : contains histogram plots, z profiling plots, multiple scan plots, etc.
- co2_postprocessing.py : script specifically to determine porosity and $CO_2$ saturation along the core

Examples/Tutorials:
- EX1: How to read dicom files, untiltening and registering
- EX2: How to pre- and postprocess CT data to estimate $CO_2$ saturation along a core sample

### Data availability
If you need access to the examples data, please contact Dr. David DiCarlo at dicarlo@mail.utexas.edu

