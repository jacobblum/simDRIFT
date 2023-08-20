import sys
import argparse
import numba
import os 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_suite import run_tests

class CLI:
    """Command line interface for running the simDRIFT test suite
    """    
    def __init__(self, subparsers) -> None:
        """Initializes subparsers

        :param subparsers: argparse object containing of subparsers for each input parameter
        :type subparsers: argparse
        """        
        self.subparsers = subparsers
        pass

    def add_subparser_args(self):
        """Adds subparser for the ``run_tests`` command.

        :return: argparse object containing of subparsers for each input parameter
        :rtype: argparse
        """        
        subparser = self.subparsers.add_parser("run_tests",
                                        description="run the simDRIFT test suite"
                                        )
        return self.subparsers
    
    def validate_args(self, args):
        """Function to validate input arguments for testing

        :param args: input arguments (Note, ``run_tests`` accepts no arguments)
        :type args: str
        """        
        assert len(args) < 1, "run_tests accepts no arguments"
    
    def run(self, args):
        """Passes to the test suite file

        :param args: input arguments (Note, ``run_tests`` accepts no arguments)
        :type args: str
        """        
        run_tests()
