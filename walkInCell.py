import numpy as np 


def walk_in_cell(spin, cell_info, fiber_xycordinate, TE, dt, spin_pos):    
    # Params
    spin_trajectories = np.zeros((int(TE/dt),3))
    spin_trajectories[0,:] = spin
    D = 3.0
    step = np.sqrt(6*D*dt)
   
    # Find Fibers Within A Cell
    cell_x, cell_y, cell_z, radius = cell_info[0], cell_info[1], cell_info[2], cell_info[3]
    fibers_xy = fiber_xycordinate[fiber_xycordinate[:,3] == 0]
    fibers_yz = fiber_xycordinate[fiber_xycordinate[:,3] == 1]

    fibers_xy_nrby = fibers_xy[(((cell_x) - radius < fibers_xy[:,0]) & (fibers_xy[:,0] < cell_x + radius)) & ((cell_y - radius < fibers_xy[:,1]) & (fibers_xy[:,1] < cell_y + radius)), :]
    fibers_yz_nrby = fibers_yz[(((cell_z) - radius < fibers_yz[:,0]) & (fibers_yz[:,0] < cell_z + radius)) & ((cell_y - radius < fibers_yz[:,1]) & (fibers_yz[:,1] < cell_y + radius)), :]



    for i in (range(1, int(TE/dt))):
        dist_from_cell_origin = cell_info[3] + 0.01
        while(dist_from_cell_origin > cell_info[3]):
            dir = np.random.uniform(-1,1,3)
            dir = dir / np.sqrt(np.sum(dir**2))
            spin_trajectories[i,:] = spin_trajectories[i-1,:] + step * dir
            dist_from_cell_origin = np.linalg.norm( spin_trajectories[i,0:3] - cell_info[0:3], ord = 2)
            if any([np.linalg.norm(spin_trajectories[i,0:2]-fibers_xy_nrby[j,0:2], ord = 2) < fibers_xy_nrby[j,5] for j in range(fibers_xy_nrby.shape[0])]):
                dist_from_cell_origin = cell_info[3] + 0.01
            dist_yz_vector = np.zeros((fibers_yz_nrby.shape[0],2))
            dist_yz_vector[:,0] = fibers_yz_nrby[:,1]
            dist_yz_vector[:,1] = fibers_yz_nrby[:,0]
            if any([np.linalg.norm(spin_trajectories[i,1:3]-dist_yz_vector[j,:], ord = 2) < fibers_yz_nrby[j,5] for j in range(fibers_yz_nrby.shape[0])]):
                dist_from_cell_origin = cell_info[3] + 0.01
    
    
    position = np.array([[0,0,spin_pos]])
    spin_trajectories = np.append(spin_trajectories, position, axis= 0)
         
    return spin_trajectories 