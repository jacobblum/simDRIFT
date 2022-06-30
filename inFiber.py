import numpy as np 

"""
Module Deterimins if a Spin is within a Fiber in some neighborhood, nbdh 
"""


def in_fiber(spin, fiber_xycordinate,nbhd, Crossing):
    fibers_xy = fiber_xycordinate[fiber_xycordinate[:,3] == 0]
    fibers_yz = fiber_xycordinate[fiber_xycordinate[:,3] == 1]
    output_arg = 0
    fiber_info = np.zeros(fibers_xy.shape[1])
    fiber_in_xy = fibers_xy[np.where(np.array([np.linalg.norm(spin[0:2] - fibers_xy[i,0:2], ord = 2) < fibers_xy[i,5] for i in range(fibers_xy.shape[0])]) == True)]

    yz_dist_vec = np.zeros((fibers_yz.shape[0],2))
    yz_dist_vec[:,0] = fibers_yz[:,1]
    yz_dist_vec[:,1] = fibers_yz[:,0]

    fiber_in_yz = fibers_yz[np.where(np.array([np.linalg.norm(spin[1:3]- yz_dist_vec[i,:], ord = 2) < fibers_yz[i,5] for i in range(fibers_yz.shape[0])]) == True)]

    if fiber_in_xy.shape[0] > 0:
        fiber_info = fiber_in_xy
        for i in range(fiber_info.shape[0]):
            if fiber_info[i,0] < 100:
                output_arg = 4
            elif fiber_info[i,0] >= 100:
                output_arg = 1
    
    if fiber_in_yz.shape[0] > 0:
        output_arg = 2
        fiber_info = fiber_in_yz




    return output_arg, fiber_info



