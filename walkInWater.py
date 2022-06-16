import numpy as np


def walk_in_water(spin, fiber_xycordinate, cell_centers, TE, dt, spin_loc):
    spin_trajectory = np.zeros((int(TE/dt),3))
    D = 5.0
    step = np.sqrt(6*D*dt)
    spin_trajectory[0,:] = spin

    fibers_xy = fiber_xycordinate[fiber_xycordinate[:,3] == 0]
    fibers_yz = fiber_xycordinate[fiber_xycordinate[:,3] == 1]

    dist_yz_vector = np.zeros((fibers_yz.shape[0],2))
    dist_yz_vector[:,0] = fibers_yz[:,1]
    dist_yz_vector[:,1] = fibers_yz[:,0] 
    
    for i in (range(1, int(TE/dt))):
        k = 0
        invalid_move = True
        while(invalid_move):
            k += 1
            dir = np.random.uniform(-1,1,3)
            dir = dir/(np.sqrt(  (dir[0])**2 + (dir[1])**2 + (dir[2])**2))
            spin_trajectory[i,:] = spin_trajectory[i-1,:] + step * dir
            not_in_a_cell = True
            not_in_a_v_fiber = True
            not_in_a_h_fiber = True  
            # Find Cell Perimeters Within 1 Step Size of the Current Position


            nrby = np.array([np.linalg.norm(spin_trajectory[i-1, :] - cell_centers[j,0:3], ord = 2) - cell_centers[j,3] < step for j in range(cell_centers.shape[0])])
            cells_nrby = cell_centers[np.where(nrby == True)]
       
            # Find Fiber Perimeters Within 1 Step Size of the Current Position
            nrby_xy_idx = np.array([np.linalg.norm(spin_trajectory[i-1,0:2] - fibers_xy[j,0:2], ord = 2) - fibers_xy[j,5] < step for j in range(fibers_xy.shape[0])])
            fibers_xy_nrby = fibers_xy[np.where(nrby_xy_idx == True)]
            nrby_yz_idx = np.array([np.linalg.norm(spin_trajectory[i-1,1:3] - dist_yz_vector[j,:], ord = 2) - fibers_yz[j,5] < step for j in range(fibers_yz.shape[0])])
            fibers_yz_nrby = fibers_yz[np.where(nrby_yz_idx == True)]

            if fibers_yz_nrby.shape[0] > 0:
                fibers_yz_nrby_dist = np.zeros((fibers_yz_nrby.shape[0],2))
                fibers_yz_nrby_dist[:,0] = fibers_yz_nrby[:,1]
                fibers_yz_nrby_dist[:,1] = fibers_yz_nrby[:,0]

            if any([np.linalg.norm(spin_trajectory[i,0:3] - cells_nrby[j,0:3]) < cells_nrby[j,3] for j in range(cells_nrby.shape[0])]):
                not_in_a_cell = False
            if any([np.linalg.norm( spin_trajectory[i,0:2] - fibers_xy_nrby[j,0:2], ord = 2) < fibers_xy_nrby[j,5] for j in range(fibers_xy_nrby.shape[0])]):
                not_in_a_v_fiber = False
            if any([np.linalg.norm(spin_trajectory[i,1:3]-fibers_yz_nrby_dist[j,:], ord = 2) < fibers_yz_nrby[j,5] for j in range(fibers_yz_nrby.shape[0])]):
                not_in_a_h_fiber = False
            if not_in_a_cell and not_in_a_v_fiber and not_in_a_h_fiber:
                invalid_move = False
            if k > 10000:
                print('stuck, %s' %spin_trajectory[i,:])
                break 
    position = np.array([[0,0,spin_loc]])

    spin_trajectory = np.append(spin_trajectory, position, axis= 0)
    
    return spin_trajectory
