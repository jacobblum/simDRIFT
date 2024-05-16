
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


import time



from datetime import datetime
from _tests import (
            signal_types,
            trajectory_types,
            signal_shapes,
            custom_diffusion_scheme,
            trajectory_shapes,
            water_physics,
            fiber_physics,
            cell_physics,
            single_cell_physics
        )


def run_tests():

    log_file = os.path.join(os.getcwd() + os.sep + 'test_suite_log')
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('numpy').setLevel(logging.WARNING)
    logging.basicConfig(level = logging.INFO,
                        format = 'simDRIFT: %(message)s',
                        filename = log_file,
                        filemode = 'w')
    console = logging.StreamHandler()
    formatter = logging.Formatter("simDRIFT: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info(' ... running the simDRIFT test suite ... ')
    logging.info(' The test suite will run 20 dMRI forward simulations to verify the simDRIFTs physics, output types, and shapes. On an RTX3090, the test suite takes about 8 minutes to complete; however, this time will depend on your hardware.')
    
    current_time = datetime.now().strftime('%m%d%Y_%H%M%S')

    test_suite_results_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{current_time}_test_results')
    
    logging.info(f' All results will be saved under the path: {test_suite_results_directory}')

    if not os.path.exists(test_suite_results_directory): os.mkdir(test_suite_results_directory)

    signal_types.run(test_suite_results_directory)
    trajectory_types.run(test_suite_results_directory)
    signal_shapes.run(test_suite_results_directory)
    custom_diffusion_scheme.run(test_suite_results_directory)
    trajectory_shapes.run(test_suite_results_directory)
    water_physics.run(test_suite_results_directory)
    fiber_physics.run(test_suite_results_directory)
    cell_physics.run(test_suite_results_directory)
    single_cell_physics.run(test_suite_results_directory)
