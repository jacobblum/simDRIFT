import argparse

def add_subparser_args(subparsers : argparse) -> argparse:
    """Add tool-specific arguments for simulate
    
    Args:
        subparsers: Parser object before addition of arugments specific to 
            simulate 
    
    Returns:
        parser : Parser object with additonal parameters
    
    """

    subparser = subparsers.add_parser("src",
                                      description="Simulate dMRI experiment"
                                                   "on custom defined biological domain.",
                                      help = "Forward Simulate dMRI experiment "
                                             " from the input data supplied in the "
                                             " configuration.ini file.",
                                      formatter_class = argparse.ArgumentDefaultsHelpFormatter
                                    )
    subparser.add_argument("--cfg", nargs = None, type = str, 
                           dest = 'cfg_path', 
                           required = True, 
                           help = "Please enter the path to the simulations " 
                                  "configuration file (the file must have .ini extension).")
    
    subparser.add_argument("--cuda",
                        dest="use_cuda", action="store_true",
                        help="Including the flag --cuda will run the "
                            "inference on a GPU.")
    
    subparser.add_argument("--multiple-cpus",
                           dest="use_multiprocessing_diffusion", action="store_true",
                           default=False,
                           help="Including the flag --multiple-cpu will "
                                "use more than one CPU to compute the spin's "
                                "random walk in parallel.")
    
    subparser.add_argument("--cpu-cores",
                           type=int, default=None,
                           dest="n_cores",
                           help="Number of threads to use when pytorch is run "
                                "on CPU. Defaults to the number of logical cores -2.")
    
    subparser.add_argument("--random_state",
                           type=int, default=None,
                           dest="random_state",
                           help="The random state to be used by PyTorch")
    
    return subparsers
