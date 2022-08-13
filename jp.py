from concurrent.futures import thread
from operator import xor
import numpy as np 
import numba 
from numba import jit, cuda
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math

from torch import device





@cuda.jit(device = True)
def randomDirection(rng_states, an_array, thread_id):
    sum = 0
    for i in range(an_array.shape[0]):
        an_array[i] = xoroshiro128p_normal_float32(rng_states, thread_id)
        sum = sum + math.pow(an_array[i], 2)
    for i in range(an_array.shape[0]):
        an_array[i] = an_array[i]/math.sqrt(sum)
    return an_array


@cuda.jit(device = True)
def euclidean_distance(an_array1, an_array2, rotation_reference, fiber):
    dist = 0
    proj = 0
    
    if fiber == 'fiber':  
        for i in range(an_array1.shape[0]):
            dist += math.pow(an_array1[i]-an_array2[i],2)
            proj += ((an_array1[i] - an_array2[i])*rotation_reference[i])    
        return math.sqrt(dist - math.pow(proj,2))
    if fiber == 'cell':
        for i in range(an_array1.shape[0]):
            dist += math.pow(an_array1[i]-an_array2[i],2)
        return math.sqrt(dist)

@cuda.jit(device = True)
def test_func(an_array):
    for i in range(an_array.shape[0]):
        for j in range(an_array.shape[1]):
            k = 0.0
