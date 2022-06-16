import numpy as np 


def walk_in_fiber(spin, fiber_info, TE, dt, spin_loc):
    
    spin_trajectory = np.zeros((int(TE/dt), 3))
    D = fiber_info[4]
    step = np.sqrt(6*D*dt)
    spin_trajectory[0,:] = spin
    fiber_radius = fiber_info[5]
    for i in (range(1,int(TE/dt))):
        dist = fiber_radius+0.01
        b = 0
        while(dist > fiber_radius):
            b += 1
            dir = np.random.uniform(-1,1, size = 3)
            dir = dir/np.sqrt(dir[0]**2 + dir[1]**2 + dir[2]**2)
            spin_trajectory[i,:] = spin_trajectory[i-1,:] + step * dir 
            if fiber_info[3] == 0:  #Vertical Fiber Position
                dist = np.linalg.norm(spin_trajectory[i,0:2] - fiber_info[0:2], ord = 2)
            if fiber_info[3] == 1: # Horitontal Fiber Position
                f = np.array([fiber_info[1], fiber_info[0]])
                dist = np.linalg.norm(spin_trajectory[i,1:3]-f, ord = 2)
            if b > 1000:
                break
            
    position = np.array([[0,0,spin_loc]])
    spin_trajectory = np.append(spin_trajectory, position, axis = 0)
    return spin_trajectory
