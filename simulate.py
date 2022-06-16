import numpy as np
import walkInCell
import walkInFiber
import walkInWater
import sys


def Parallel(args):
    i, spins, spin_loc_key = args[0], args[1], args[2]
    spin_in_cell_info, spin_in_fiber_info, fiber_xycordinate = args[3], args[4], args[5]
    cell_centers, TE, dt = args[6], args[7], args[8] 
    sys.stdout.write('\r spin = {:08d}'.format(i+1))
    spin_trajectory = np.zeros((int(TE/dt) + 1,3))
    if spin_loc_key[i] == 1 or spin_loc_key[i] == 2:
            spin_trajectory[:,:] = walkInFiber.walk_in_fiber(spins[i,:], spin_in_fiber_info[i,:], TE, dt, spin_loc_key[i])
    if spin_loc_key[i] == 3:
            spin_trajectory[:,:] = walkInCell.walk_in_cell(spins[i,:], spin_in_cell_info[i,:], fiber_xycordinate, TE, dt, spin_loc_key[i])
    elif spin_loc_key[i] == 0:
            spin_trajectory[:,:] = walkInWater.walk_in_water(spins[i,:], fiber_xycordinate, cell_centers, TE, dt, spin_loc_key[i])
    return spin_trajectory