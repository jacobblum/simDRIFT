import nibabel as nb 
import pandas as pd 
import numpy as np 
import glob as glob 
import os
from datetime import datetime



def extract_data_from_subject(subject):
    proj_dir, subject_id = os.path.split(subject) 
    maps_list_dbsi = pd.Series(['fiber1_axial_map.nii', 'fiber1_radial_map.nii', 'fiber1_fraction_map.nii', 'fiber2_axial_map.nii', 'fiber2_radial_map.nii', 'fiber2_fiber_fraction_map.nii',
                    'hindered_fraction_map.nii', 'hindered_adc_map.nii', 'restricted_fraction_map.nii', 'restricted_adc_map.nii', 'water_fraction_map.nii', 'water_adc_map.nii'],
                    index = ['fiber 1 AD', 'fiber 1 RD', 'fiber 1 fraction', 'fiber 2 axial', 'fiber 2 radial', 'fiber 2 fraction', 'hindered fraction', 'hindered adc', 'restricted fraction', 'restricted adc',
                    'water fraction', 'water adc'])
    subject_results = pd.Series(dtype='object')
    subject_results['Mosiac ID'] = subject_id
    for index, file in maps_list_dbsi.iteritems():
        map = subject + os.sep + 'DHI_results_0.3_0.3_3_3' + os.sep + file
        if os.path.exists(map): 
            data = np.mean(nb.load(map).get_fdata())
            subject_results[index] = data
    return subject_results
             






maps_list_dbsi = pd.Series(['fiber1_axial_map.nii', 'fiber1_radial_map.nii', 'fiber1_fraction_map.nii', 'fiber2_axial_map.nii', 'fiber2_radial_map.nii', 'fiber2_fiber_fraction_map.nii',
                'hindered_fraction_map.nii', 'hindered_adc_map.nii', 'restricted_fraction_map.nii', 'restricted_adc_map.nii', 'water_fraction_map.nii', 'water_adc_map.nii'],
                index = ['fiber 1 AD', 'fiber 1 RD', 'fiber 1 fraction', 'fiber 2 axial', 'fiber 2 radial', 'fiber 2 fraction', 'hindered fraction', 'hindered adc', 'restricted fraction', 'restricted adc',
                'water fraction', 'water adc'])

subjects = glob.glob(r"C:\MCSIM\dMRI-MCSIM-main\6x6-Mosaic_DBSI-vs-NEO\Row*_Col*")

total_data_frame = pd.DataFrame()
for subject in subjects:
    total_data_frame = total_data_frame.append(extract_data_from_subject(subject), ignore_index = True)
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
total_data_frame.to_csv(r"C:\MCSIM\dMRI-MCSIM-main\6x6-Mosaic_DBSI-vs-NEO\Mosiac_Results_DBSI_{}.csv".format(dt_string))