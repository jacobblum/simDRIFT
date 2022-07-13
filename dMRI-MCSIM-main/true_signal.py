import numpy as np
import nibabel as nb 
import matplotlib.pyplot as plt 




def main():
    dwi = nb.load(r"C:\po1_data\diff_unwarp.nii.gz").get_fdata()
    roi = nb.load(r"C:\po1_data\signal_roi.nii.gz").get_fdata()
    bvals = np.loadtxt(r"C:\po1_data\bval").T
    dwi = dwi[roi == 1].T 
    signal = dwi * 1/np.amax(dwi)

    return signal

