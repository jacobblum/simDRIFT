import numpy as np 


def in_cell(spin, cell_centers, nbhd):
    s_x, s_y, s_z = spin[0], spin[1], spin[2]
    cell_info = np.zeros(cell_centers.shape[1])
    output_arg = 0
    k = 0
    cell_within = cell_centers[np.where(np.array([np.linalg.norm( spin - cell_centers[i,0:3], ord = 2) <= cell_centers[i,3] for i in range(cell_centers.shape[0])]) == True)]
    if cell_within.shape[0] > 0:
        cell_info = cell_within
        output_arg = 3.0
    return output_arg, cell_info