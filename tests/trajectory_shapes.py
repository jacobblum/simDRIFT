
import sys 
import logging
import os 
import numpy as np
import pytest
import nibabel as nb 
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
import glob as glob
from pathlib import Path
from src.data import diffusion_schemes

@pytest.mark.parametrize("input, expected", [(100, 100), (256 * 1e3, 256 * 1e3), (1e6,1e6,)])
def test_trajectory_shapes(input, expected):
    """1. Check that the forward simulated trajectories match the number of simulated spins
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cmd = r"simDRIFT"
    cmd += f" simulate --n_walkers {int(input)} --fiber_fractions 0,0 --cell_fractions 0,0 --Delta 0.001 --voxel_dims 10 --buffer 0 --verbose no" 
    os.system(cmd)
    trajectories = np.load(os.getcwd() + os.sep + 'trajectories' + os.sep + 'total_trajectories_t2p.npy')
    assert trajectories.shape[0] == int((expected))