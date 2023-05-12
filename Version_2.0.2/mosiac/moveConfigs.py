import contextlib
from email.policy import default
import multiprocessing as mp
from multiprocessing.sharedctypes import Value
import numpy as np 
import numba 
from numba import jit, cuda, int32, float32
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math
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
import jp as jp

def main():
    configs = glob.glob(r"C:\Users\kainen.utt\Documents\Research\MosaicProject\intraVoxelTest\cellEdge\SimOutput\*\*.ini")
    for cfg in configs:
        destination = r"C:\Users\kainen.utt\Documents\Research\MosaicProject\intraVoxelTest\cellEdge\Configs" + os.sep + str(cfg).split(r'SimOutput',1)[1][3] + str(cfg).split(r'SimOutput',1)[1][7] + "_Config.ini"
        shutil.copyfile(cfg, destination)


if __name__ == "__main__":
    main()