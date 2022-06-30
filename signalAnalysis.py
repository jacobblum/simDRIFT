import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import nibabel as nb 
import os


data = pd.Series(['dti_adc_map.nii', 'dti_axial_map.nii', 'dti_fa_map.nii', 'dti_radial_map.nii', 'fiber_fraction_map.nii', 'fiber1_fiber_fraction_map.nii',
                'fiber1_axial_map.nii', 'fiber1_fa_map.nii', 'fiber1_radial_map.nii', 'fiber2_fiber_fraction_map.nii',  'fiber2_axial_map.nii', 'fiber2_fa_map.nii', 'fiber2_radial_map.nii', 'hindered_fraction_map.nii', 'restricted_fraction_map.nii',
                'water_fraction_map.nii'],
                index = ['DTI-ADC', 'DTI-AD', 'DTI-FA', 'DTI-RD', 'FF', 'F1-FF', 'F1-AD', 'F1-FA', 'F1-RD', 'F2-FF'   ,'F2-AD','F2-FA', 'F2-RD', 'HF', 'RF', 'WF' ])


directories = pd.Series([r'/Volumes/LaCie/MCSIM-Jacob/CrossingFibers/DBSI-99/DHI_results_0.3_0.3_3_3', 
                            r'/Volumes/LaCie/MCSIM-Jacob/CrossingFibers/ABCD/DHI_results_0.3_0.3_3_3'],
                            index = [0,1])


data_matrix = np.zeros((len(directories.index), len(data.index)))

for i in range(len(directories.index)):
    for j in range(len(data.index)):
        dbsi_map = nb.load(directories[i] + os.sep + data[j]).get_fdata()
        data_matrix[i,j] = np.mean(dbsi_map)



fig, ax = plt.subplots(figsize = (10,5))

ax.imshow(data_matrix, cmap = 'Dark2')
ax.set_yticks([0,1])
ax.set_yticklabels(['99-Dir', 'ABCD'])
ax.set_xticks(np.arange(0,len(data.index), 1))
ax.set_xticklabels(data.index)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")


for i in range(len(directories.index)):
    for j in range(len(data.index)):
        text = ax.text(j, i, round(data_matrix[i, j], 3),
                       ha="center", va="center", color="black")
ax.set_title('Comparison of ABCD-DBSI Diffusion Scheme :no extra Signals')
plt.show()
