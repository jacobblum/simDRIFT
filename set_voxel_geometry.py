from cProfile import label
from pickletools import read_unicodestring1
import numpy as np 
import numba 
from numba import jit, cuda, int32, float32
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math
import jp
import matplotlib.pyplot as plt
import time 
import os  
from tqdm import tqdm
import nibabel as nb
import glob as glob 
import configparser
from ast import literal_eval
from multiprocessing import Process
import shutil
import walk_in_fiber
import walk_in_cell
import walk_in_extra_environ
import spin_init_positions
import sys
import diffusion


def set_num_fibers(fiber_fractions, fiber_radius, voxel_dimensions, buffer):
    num_fibers = []
    for i in range(len(fiber_fractions)):
        num_fiber = int(np.sqrt((fiber_fractions[i] * (voxel_dimensions + buffer)**2)/(np.pi*fiber_radius**2)))
        num_fibers.append(num_fiber)    
    sys.stdout.write('\nplacing {} fiber grid...'.format(num_fibers))
    return num_fibers

