from concurrent.futures import thread
from operator import xor
import numpy as np 
import numba 
from numba import jit, cuda
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math

@cuda.jit(device = True,nopython = False)
def random_on_S2(rng_states, an_array, thread_id):
    sum = 0
    for i in range(an_array.shape[0]):
        an_array[i] = xoroshiro128p_normal_float32(rng_states, thread_id)
        sum = sum + math.pow(an_array[i], 2)
    for i in range(an_array.shape[0]):
        an_array[i] = an_array[i]/math.sqrt(sum)
    return an_array


    
