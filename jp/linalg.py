from concurrent.futures import thread
from operator import xor
import numpy as np 
import numba 
from numba import jit, cuda
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math
import sys
import logging


def Ry(thetas):
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
   
    rotation_matricies = np.zeros((len(thetas),3,3))
    for i, theta in enumerate(thetas):

        theta = np.radians(float(theta))
        s, c = np.sin(theta), np.cos(theta)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        rotation_matricies[i,:,:] = Ry
    logging.info('------------------------------')
    logging.info('Rotation Matrices ')
    logging.info('------------------------------')
    logging.info('\n{}'.format(rotation_matricies[0,:,:]))
    logging.info('\n{}'.format(rotation_matricies[1,:,:]))
    return rotation_matricies

def affine_transformation(xv: np.ndarray, x: float, y: float, thetas, i):
    
    if xv.shape[0] % 2 == 1:
        middle_fiber_index = (xv.shape[0]-1) //2 
    elif xv.shape[0] % 2 == 0:
        middle_fiber_index = (xv.shape[0]) // 2
    
    """ Align the middle (in x) of the fiber bundles """

    dM_x = np.cos(np.deg2rad(thetas[0]))*xv[middle_fiber_index,0] - np.cos(np.deg2rad(thetas[i]))*xv[middle_fiber_index,0] 
    dM_z = np.sin(np.deg2rad(thetas[i]))*xv[middle_fiber_index,0] - np.sin(np.deg2rad(thetas[0]))*xv[middle_fiber_index,0]
    dZ   = np.sin(np.deg2rad(thetas[0]))*xv[middle_fiber_index,0] 

    Ax  = np.array([np.cos(np.deg2rad(thetas[i]))*x, y, -np.sin(np.deg2rad(thetas[i]))*x])  
    b   = np.array([dM_x, 0., dM_z + dZ])
 
    return Ax + b


@cuda.jit(device = True)
def dL2(x,y,v, project_along):
    dist = 0
    proj = 0

    if project_along:
        for i in range(x.shape[0]):
            dist += math.pow(x[i]-y[i],2)
            proj += ((x[i] - y[i])*v[i])    
        return math.sqrt(dist - math.pow(proj,2))
    
    else:
        for i in range(x.shape[0]):
            dist += math.pow(x[i]-y[i],2)
        return math.sqrt(dist)

