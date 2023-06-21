import sys
import argparse
import numba
import os 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_suite import run_tests

class CLI:
    def __init__(self, subparsers) -> None:
        self.subparsers = subparsers
        pass

    def add_subparser_args(self):
        subparser = self.subparsers.add_parser("run_tests",
                                        description="run the simDRIFT test suite"
                                        )
        return self.subparsers
    
    def validate_args(self, args):
        assert len(args) < 1, "run_tests accepts no arguments"
    
    def run(self, args):
        run_tests()
