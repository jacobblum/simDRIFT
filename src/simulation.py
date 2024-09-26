import os
import glob as glob
from datetime import datetime
import sys
import src.physics.diffusion as diffusion
import src.save as save 
from src.setup import set_voxel_configuration
import logging
from typing import Dict, Type
from . import gradients 
import torch
import argparse

logger = logging.getLogger('simDRIFT')

def run_simulation(args : Type[argparse.Namespace]) -> None:
    """ The full script for the command line tool to perform the spin ensemble's random walk on the tissue micro-structure 
        induced by the configuration file
    
    Args:
        args: Inputs from the command line, already parsed using argparse
    
    Note: Returns nothing, but writes output to file(s) specified from 
        command line.
    
    """
    # log the start time
    logger.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.info('Running simulate.')

    # handle the random state
    RANDOM_SEED = args.random_state or torch.randint(low = 0, high = 10000, size = (1,)).item()
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed = RANDOM_SEED)
        logger.info(f"PyTorch random seed set to : {RANDOM_SEED}")
    else:
        torch.manual_seed(RANDOM_SEED)

    try:
        # Load and Create Gradient Scheme
        gradient_obj = gradients.get_gradient_obj(args)

        # Create The Simulation Geometry and Save it.
        voxel_geometry_obj = set_voxel_configuration.instantiate_geometry_obj(args)

        # Execute the Simulation
        ensemble_phase_data = diffusion.compute_random_walk(
            geometry_obj=voxel_geometry_obj,
            gradient_obj=gradient_obj,
            args = args,
        )
        # Save the data
        save.calc_and_save_signals(ensemble_phase_data)

        logger.info('Completed simulate.')
        logger.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    except KeyboardInterrupt:
        logger.info('Keyboard interrupt. Terminated without saving.')
        sys.exit(1)

    return
