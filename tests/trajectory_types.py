
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


@pytest.mark.parametrize("expected", [('.npy'),])
def test_trajectory_types(expected):
    """1. Check that the forward simulated trajectory is a .npy file
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cmd = r"simDRIFT"
    cmd += f" simulate --n_walkers 256000 --fiber_fractions 0,0 --cell_fractions 0,0 --Delta .001 --water_diffusivity {input} --voxel_dims 10 --buffer 0 --verbose no" 
    os.system(cmd)
    trajectories = glob.glob(os.getcwd() + os.sep + 'trajectories' + os.sep + '*')
    for trajectory in trajectories:
        assert os.path.splitext(trajectory)[-1:] == (expected,)
