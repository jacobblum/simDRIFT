from turtle import pos
from matplotlib.pyplot import step
import numpy as np 
import numba 
from numba import jit, cuda
from numba.cuda import random 
 
@cuda.jit 
def random_walk_gpu(init_position, positions, dir, T, dt):    
    x_ctr = 0.0
    y_ctr = 0.0
    circle_radius = 1.0
    step_size = .01 
    dist = circle_radius + step_size
    positions[0,:] = init_position

    for i in range(1, int(T/dt)):
        dist = circle_radius + step_size
        while(dist > circle_radius):
            for k in range(3):
                dir[k] = (-1-1)*np.random.rand() -1 
            dir = dir/np.linalg.norm(dir, ord = 2)
            positions[i, :] = positions[i-1] + step_size * dt
        


T = 1
dt = .0050
positions = np.zeros((int(T/dt),3))
init_position = np.zeros(2, np.int32)
dir = np.zeros(3, np.int32)
random_walk_gpu[1,1](init_position, positions, dir, T, dt)

print(np.random.uniform(0,1))