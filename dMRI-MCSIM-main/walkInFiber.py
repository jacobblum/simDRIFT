import numpy as np 


def walk_in_fiber(spin, fiber_info, TE, dt, spin_loc):
    spin_trajectory = np.zeros((int(TE/dt), 3))
    D = fiber_info[4]
    step = np.sqrt(6*D*dt)
    spin_trajectory[0,:] = spin
    fiber_radius = fiber_info[5]
    for i in (range(1,int(TE/dt))):
        dir = np.random.uniform(-1,1, size = 3)
        u = dir/np.linalg.norm(dir, ord = 2)
        spin_trajectory[i,:] = spin_trajectory[i-1,:] + step*u 
        # Checks xy-distance for vertical fiber 
        if fiber_info[3] == 0:
            dist = np.linalg.norm(spin_trajectory[i,0:2] - fiber_info[0:2], ord = 2)
        # Checks yz-distance for horizontal fiber 
        if fiber_info[3] == 1: 
            f = np.array([fiber_info[1], fiber_info[0]])
            dist = np.linalg.norm(spin_trajectory[i,1:3]-f, ord = 2)  
    
        # Logic for valid stepping 
        if dist > fiber_radius:
            spin_trajectory[i,:] = spin_trajectory[i-1,:]
        else:
            spin_trajectory[i,:] = spin_trajectory[i,:]


    position = np.array([[0,0,spin_loc]])
    spin_trajectory = np.append(spin_trajectory, position, axis = 0)
    return spin_trajectory
