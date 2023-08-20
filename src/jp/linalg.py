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
    """Calculates rotation matrices for the desired fiber orientations

    :param thetas: List of angles (with respect to the `y`-axis) for each fiber bundle
    :type thetas: int, tuple
    :return: Rotation matrices for each fiber bundle.
    :rtype: np.ndarray
    """ 
    logging.info('------------------------------')
    logging.info(' Rotation Matrices ')
    logging.info('------------------------------')
   
    rotation_matricies = np.zeros((len(thetas),3,3))
    for i, theta in enumerate(thetas):
        theta = np.radians(float(theta))
        s, c = np.sin(theta), np.cos(theta)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        rotation_matricies[i,:,:] = Ry
        logging.info(' [[{: .2f}, {: .2f}, {: .2f}],'.format(Ry[0,0],Ry[0,1],Ry[0,2]))
        logging.info('  [{: .2f}, {: .2f}, {: .2f}],'.format(Ry[1,0],Ry[1,1],Ry[1,2]))
        logging.info('  [{: .2f}, {: .2f}, {: .2f}]]\n'.format(Ry[2,0],Ry[2,1],Ry[2,2]))

    return rotation_matricies

def affine_transformation(xv: np.ndarray, x: float, y: float, thetas, i):
    """Calculates and applies affine transformation to fiber grid.

    :param xv: Fiber grid
    :type xv: np.ndarray
    :param x: `x` coordinates
    :type x: float
    :param y: `y` coordinates
    :type y: float
    :param thetas: _description_
    :type thetas: int, tuple
    :param i: bundle index
    :type i: int
    :return: Transformed coordinates
    :rtype: np.ndarray
    """
    
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


@cuda.jit(device = True,nopython = False)
def dL2(x,y,v, project_along):
    """Internal linear algebra function for projecting along transformed vectors.

    :param x: `x` coordinate
    :type x: float
    :param y: `x` coordinate
    :type y: float
    :param v: Vector to project along
    :type v: list
    
    :return: Distance
    :rtype: float
    """
    
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

