import sys
sys.path.append("..")
sys.path.append(".")
import argparse

from master_cli import AbstractCLI
import os 
import src.cfgparser as cfgparser
import torch
from typing import Type
import logging
from datetime import datetime
from src import simulation
import psutil

class CLI(AbstractCLI):
  
    def __init__(self):

        self.name = "simulate"  
        self.args = None

    def get_name(self) -> str:
        return self.name

    @staticmethod
    def validate_args(args) -> argparse.Namespace:   
        """Validate parsed arguments"""

        # Ensure that the cfg file is accessible
        assert os.access(args.cfg_path, os.R_OK), \
            f"Cannot read specified simulation configuration file {args.cfg_path}. " \
            f"Ensure that the file exists and is read accessible."
        
        # Parse the configuration file
        cfg_args = cfgparser.parse_config_file(args.cfg_path)

        # Unify parsed configuration file args with argparse.Namespace object
        for k,v in cfg_args.items(): setattr(args, k, v)

        # If cuda is flagged, make sure it is availalbe.
        if args.use_cuda:
            assert torch.cuda.is_available(), \
            f"Trying to use CUDA, but the device is not availalbe"
        
        else:
            # Warn the user in case the CUDA flag was forgotten by mistake.
            if torch.cuda.is_available():
                sys.stdout.write("Warning: CUDA is available, but will not be "
                                 "used.  Use the flag --cuda for "
                                 "significant speed-ups.\n\n")
                sys.stdout.flush()  # Write immediately

        if args.use_multiprocessing_diffusion:
            assert ((args.n_cores >= 1) and (args.n_cores) <= psutil.cpu_count()), \
                f"--cpu-cores must be an integer >= 1"

        # If user specifies cpu and gpu parallelizataion, default to 
        # the gpu.
        if ((args.use_cuda) and (args.use_multiprocessing_diffusion)):
            args.use_multiprocessing_diffusion = False      

        return args   

    @staticmethod
    def run(args):
        # Run the tool
        return main(args)

def setup_and_logging(args) -> None:
    simulation_outputs_base_dir = os.path.join(
                                              args.output_directory,
                                              f"{datetime.now().strftime('%Y%m%d_%H%M')}_simulation_results"
                                              )
    # if the created directory does not exist 
    if not os.path.exists(simulation_outputs_base_dir): 
        os.mkdir(simulation_outputs_base_dir)          

        assert os.access(simulation_outputs_base_dir, os.W_OK), \
            f"Make sure propper permisions are configured to write \n"\
            f"simulation output data to {simulation_outputs_base_dir}"
        
    setattr(args, 'sim_out_dir', simulation_outputs_base_dir)

    log_file = os.path.join(args.sim_out_dir, "log.log")
    logger = logging.getLogger('simDRIFT') # Name of the logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("simDRIFT:simulate: %(message)s")
    file_handler = logging.FileHandler(filename=log_file, mode='w', encoding='UTF-8')
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)  # set the file format
    console_handler.setFormatter(formatter)  # use the same format for stdout
    logger.addHandler(file_handler)  # log to file
    logger.addHandler(console_handler)

    # Log the command as typed by user

    logger.info("Command:\n"
            + ' '.join(['simDRIFT', 'simulate'] + sys.argv[2:]))

    return args, file_handler
    
def main(args) -> None:
    args, file_handler = setup_and_logging(args)
    
    # Run the tool.
    simulation.run_simulation(args)
    file_handler.close()


    return 