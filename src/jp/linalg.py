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
        logging.info(' [[{: .2f}, {: .2f}, {: .2f}],'.format(   Ry[0,0],Ry[0,1],Ry[0,2]))
        logging.info('  [{: .2f}, {: .2f}, {: .2f}],'.format(   Ry[1,0],Ry[1,1],Ry[1,2]))
        logging.info('  [{: .2f}, {: .2f}, {: .2f}]]\n'.format( Ry[2,0],Ry[2,1],Ry[2,2]))

    return rotation_matricies

def affine_transformation(xv_total: np.ndarray, yv_total : np.ndarray, x: float, y: float, thetas, i):
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



    current_bundle_index = i
    xv           = xv_total[i, ...]
    yv           = yv_total[i, ...]
    
    ### If current_bundle_index = 0 -> register into first octant!, else -> go to current_bundle_index - 1 frame
    
    if xv.shape[0] % 2 == 1:   middle_fiber_index = (xv.shape[0]-1) //2 
    elif xv.shape[0] % 2 == 0: middle_fiber_index = (xv.shape[0]) // 2

    if current_bundle_index == 0:
        dM_x = xv[middle_fiber_index,0] - np.cos(np.deg2rad(thetas[i]))*xv[middle_fiber_index,0] 
        dM_z = np.sin(np.deg2rad(thetas[i]))*xv[middle_fiber_index,0] - xv[middle_fiber_index,0]
        dZ   = xv[middle_fiber_index,0] 
    else:
        dM_x = np.cos(np.deg2rad(thetas[0]))*xv[middle_fiber_index,0] - np.cos(np.deg2rad(thetas[i]))*xv[middle_fiber_index,0] 
        dM_z = np.sin(np.deg2rad(thetas[i]))*xv[middle_fiber_index,0] - np.sin(np.deg2rad(thetas[0]))*xv[middle_fiber_index,0]
        dZ   = np.sin(np.deg2rad(thetas[0]))*xv[middle_fiber_index,0] 
    Ax  = np.array([np.cos(np.deg2rad(thetas[i]))*x, y, -np.sin(np.deg2rad(thetas[i]))*x])  
    b   = np.array([dM_x, 0., dM_z + dZ])

    return Ax + b


@cuda.jit(device = True,nopython = False)
def dL2(x,y,v,project_along):
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

@cuda.jit(device = True, nopython = False)
def cosine_similarity(a, b):
    theta = math.acos(dot(a,b))
    return theta

@cuda.jit(device = True,nopython = False)
def dot(a, b):
    t = 0.
    for i in range(a.shape[0]):
        t += a[i]*b[i]
    return t

@cuda.jit(device = True,nopython = False)
def matmul(A, b):
    c = cuda.local.array(shape = A.shape[0], dtype = numba.float32)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            c[i] += A[i,j]*b[j]
    return c

@cuda.jit(device = True,nopython = False)
def gamma(a, b, theta, ta, cp):
    r"""evaluate the space curve, gamma, that traces the fiber.
    :param a: the proposed spin position 
    :type  a: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param b: the fiber's principle orientation (Ry(theta).dot([0., 0., 1.]))
    :type  b: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param theta: The orientation of the fiber bundle 
    :type  theta: float32
    :param ta: memory to write the dynamic fiber center to 
    :type  ta: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param cp: the curvature parameters of the fiber; cp[0] = kappa, cp[1] = L, cp[2] = A, cp[3] = P
    :type  ta: numba.cuda.cudadrv.devicearray.DeviceNDArray
    
    :return: ta
    :rtype: numba.cuda.cudadrv.devicearray.DeviceNDArray
    """
    t     = dot(a,b)
    x = cp[2] * math.sin(math.pi*cp[0] /((1/cp[3])*cp[1])*t)
    y = numba.float32(0.0)
    z = numba.float32(t)
    
    ta[0] =  math.cos(theta)*x + math.sin(theta)*z
    ta[1] =  y
    ta[2] = -math.sin(theta)*x + math.cos(theta)*z

    return ta

@cuda.jit(device = True,nopython = False)
def d_gamma__d_t(a, b, theta, ta, cp):
        
    r"""calculate the unit-normal tangent vector to the space space curve, gamma, that traces the fiber.
    :param a: the proposed spin position 
    :type  a: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param b: the fiber's principle orientation (Ry(theta).dot([0., 0., 1.]))
    :type  b: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param theta: The orientation of the fiber bundle 
    :type  theta: float32
    :param ta: memory to write the dynamic fiber center to 
    :type  ta: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param cp: the curvature parameters of the fiber; cp[0] = kappa, cp[1] = L, cp[2] = A, cp[3] = P
    :type  ta: numba.cuda.cudadrv.devicearray.DeviceNDArray
    
    :return: ta
    :rtype: numba.cuda.cudadrv.devicearray.DeviceNDArray
    """

    t = dot(a,b)

    x =  cp[2] * (math.pi*cp[0] /((1/cp[3])*cp[1])) * math.cos(math.pi*cp[0] /((1/cp[3])*cp[1])*t)
    y = numba.float32(0.0)
    z = numba.float32(1.0)

    l2_norm = math.sqrt(math.pow(x, 2) + math.pow(y,2) + math.pow(z, 2))

    x = x / l2_norm
    y = y / l2_norm
    z = z / l2_norm
    
    ta[0] = math.cos(theta)*x + math.sin(theta)*z
    ta[1] = y
    ta[2] = -math.sin(theta)*x + math.cos(theta)*z
    
    return ta


