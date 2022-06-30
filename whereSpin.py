import numpy as np 
import inCell
import inFiber
from tqdm import tqdm

def where_spin(fiber_xycordinate, cell_centers, spins, Crossing):
    b = 0
    spin_loc_key = np.zeros(len(spins[:,0]))
    spin_in_fiber_centers = np.zeros((spins.shape[0],fiber_xycordinate.shape[1]))
    spin_in_cell_centers = np.zeros((spins.shape[0],cell_centers.shape[1]))

    nbhd_fiber = 0.0
    nbhd_cell = 0.0

    if fiber_xycordinate.shape[0] > 0:
        nbhd_fiber = np.amax(fiber_xycordinate[:,5])
    if cell_centers.shape[0] > 0:
        nbhd_cell = np.amax(cell_centers[:,3])
    for i in tqdm(range(len(spins[:,0])), desc = 'finding spins'):
        in_fiber_num, fiber_info = (inFiber.in_fiber(spins[i,:],fiber_xycordinate, nbhd_fiber, Crossing)) # 1 = in vertical fiber, 2 = in horizontal fiber
        in_cell_num, cell_info = inCell.in_cell(spins[i,:], cell_centers, nbhd_cell) # 3 = in a cell

        if in_cell_num > 0 and in_fiber_num == 0:
            spin_loc_key[i] = in_cell_num
            spin_in_cell_centers[i,:] = cell_info 

        elif in_fiber_num > 0 and in_cell_num == 0:
            spin_loc_key[i] = in_fiber_num
            spin_in_fiber_centers[i,:] = fiber_info

        elif in_cell_num > 0 and in_fiber_num > 0:
            b += 1
            spin_loc_key[i] = in_fiber_num
            spin_in_fiber_centers[i,:] = fiber_info


    print('both: %s' %b)        
    return spin_loc_key, spin_in_fiber_centers, spin_in_cell_centers 