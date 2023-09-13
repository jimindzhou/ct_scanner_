# The University of Texas at Austin
# Hildebrand Department of Petroleum and Geosystems Engineering
## Core-Tomography data pre- and post-processing functions and examples
## Collaborators:
- Jose Ubillus | MS Student
- Almostafa Alhadi | PhD Student

### Description
The repository contains scripts with functions available for CT data file processing, image processing and tutorials on how to use the different scripts.
Function scripts:
- readers_preprocessing.py : contains dicom file reader to numpy array, create circular mask around cores, resampling scheme (voxel coarsening), etc.
- plot_scans.py : contains histogram plots, z profiling plots, multiple scan plots, etc.
- co2_postprocessing.py : script specifically to determine porosity and $CO_2$ saturation along the core
