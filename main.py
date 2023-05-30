import functions as ft 
import matplotlib.pyplot as plt
import cv2
import numpy as np
import SimpleITK as sitk
# import file path and read dicom files
dry= 'C:/Users/ubillusj/Desktop/Almostafa/N2_dry/Raw/'
wet = 'C:/Users/ubillusj/Desktop/Almostafa/100%_brine/Raw_wet/'

# read dicom files
dry_slices = ft.read_dicom(dry)
wet_slices = ft.read_dicom(wet)

# untilting
center, dry_aligned = ft.center_itk(dry_slices)

# registering
wet_reg = ft.volume_registration(dry_aligned,wet_slices)






