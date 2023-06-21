import sys 
import argparse
from abc import ABC, abstractmethod
from src.cli import CLI as simCLI
from tests.cli import CLI as testsCLI


def main():
    TOOL_DICT = {'simulate': simCLI,
                 'run_tests': testsCLI}
    
    parser = argparse.ArgumentParser(prog='cli',
                                    description='What the program does',
                                    epilog='Text at the bottom of help')
    subparsers = parser.add_subparsers(help='sub-command help')
    
    if len(sys.argv) > 1:

        cls = TOOL_DICT[sys.argv[1]](subparsers)
        subparsers = cls.add_subparser_args()
        args = cls.validate_args(vars(parser.parse_args()))
        cls.run(args)
    

if __name__ == "__main__":
    main()