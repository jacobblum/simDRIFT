import configparser
from fractions import Fraction
import os 
import numpy as np


voxInds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]

configGeoms = ['Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Interwoven', 'Interwoven', 'Interwoven', 'Interwoven', 'Interwoven', 'Interwoven', 'Interwoven', 'Interwoven', 'Interwoven', 'Penetrating', 'Penetrating', 'Penetrating', 'Penetrating', 'Penetrating', 'Penetrating', 'Penetrating', 'Penetrating', 'Penetrating', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Void', 'Penetrating', 'Penetrating', 'Penetrating', 'Penetrating', 'Penetrating', 'Penetrating', 'Penetrating', 'Penetrating', 'Penetrating', 'Interwoven', 'Interwoven', 'Interwoven', 'Interwoven', 'Interwoven', 'Interwoven', 'Interwoven', 'Interwoven', 'Interwoven']
rows = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8]
cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8]
thetas    = [(0, 90), (0, 90), (0, 90), (0, 90), (0, 90), (0, 90), (0, 90), (0, 90), (0, 90), (0, 90), (0, 90), (0, 90), (0, 90), (0, 90), (0, 90), (0, 90), (0, 90), (0, 90), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 10), (0, 20), (0, 30), (0, 40), (0, 50), (0, 60), (0, 70), (0, 80), (0, 90), (0, 10), (0, 20), (0, 30), (0, 40), (0, 50), (0, 60), (0, 70), (0, 80), (0, 90), (0, 10), (0, 20), (0, 30), (0, 40), (0, 50), (0, 60), (0, 70), (0, 80), (0, 90), (0, 45), (0, 45), (0, 45), (0, 45), (0, 45), (0, 45), (0, 45), (0, 45), (0, 45), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 45), (0, 45), (0, 45), (0, 45), (0, 45), (0, 45), (0, 45), (0, 45), (0, 45)]
fibFracs  = [(0.2, 0.2), (0.25, 0.25), (0.35, 0.35), (0.45, 0.45), (0.55, 0.55), (0.65, 0.65), (0.7, 0.7), (0.75, 0.75), (0.8, 0.8), (0.2, 0.8), (0.25, 0.75), (0.35, 0.65), (0.45, 0.55), (0.5, 0.5), (0.6, 0.4), (0.67, 0.33), (0.7, 0.3), (0.85, 0.15), (0.2, 0.8), (0.25, 0.75), (0.35, 0.65), (0.45, 0.55), (0.5, 0.5), (0.6, 0.4), (0.67, 0.33), (0.7, 0.3), (0.85, 0.15), (0.67, 0.67), (0.67, 0.67), (0.67, 0.67), (0.67, 0.67), (0.67, 0.67), (0.67, 0.67), (0.67, 0.67), (0.67, 0.67), (0.67, 0.67), (0.67, 0.67), (0.67, 0.67), (0.67, 0.67), (0.67, 0.67), (0.67, 0.67), (0.67, 0.67), (0.67, 0.67), (0.67, 0.67), (0.67, 0.67), (0.33, 0.33), (0.33, 0.33), (0.33, 0.33), (0.33, 0.33), (0.33, 0.33), (0.33, 0.33), (0.33, 0.33), (0.33, 0.33), (0.33, 0.33), (0.33, 0.33), (0.33, 0.33), (0.33, 0.33), (0.33, 0.33), (0.33, 0.33), (0.33, 0.33), (0.33, 0.33), (0.33, 0.33), (0.33, 0.33), (0.4, 0.4), (0.4, 0.4), (0.4, 0.4), (0.4, 0.4), (0.4, 0.4), (0.4, 0.4), (0.4, 0.4), (0.4, 0.4), (0.4, 0.4), (0.25, 0.25), (0.3, 0.3), (0.35, 0.35), (0.45, 0.45), (0.5, 0.5), (0.55, 0.55), (0.6, 0.6), (0.65, 0.65), (0.7, 0.7)]
cellFracs = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.075, 0.1, 0.125, 0.15, 0.16, 0.17, 0.175, 0.18, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
cellRads  = [(2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (2.5, 2.5), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (4, 3), (4, 3), (4, 3), (4, 3), (4, 3), (4, 3), (4, 3), (4, 3), (4, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (2.5, 5), (2.5, 5), (2.5, 5), (2.5, 5), (2.5, 5), (2.5, 5), (2.5, 5), (2.5, 5), (2.5, 5)]

for iFile in voxInds:
        config = configparser.ConfigParser()
        cfgfile = open(r'/bmr207/nmrgrp/nmr202/MCSIM/input9x9/' + os.sep + "{}{}_Config.ini".format(rows[iFile], cols[iFile]), 'w')
        if cellFracs[iFile] == 0:
                simCell = False
        else:
                simCell = True

        config['Simulation Parameters'] = {
                        'numSpins': 500*10**3,
                        'fiberFraction': fibFracs[iFile],
                        'fiberRadius': 1.0,
                        'Thetas': thetas[iFile],
                        'fiberDiffusions': (1.0, 2.0),
                        'cellFraction': cellFracs[iFile],
                        'cellRadii': cellRads[iFile],
                        'fiberConfiguration': configGeoms[iFile],
                        'simulateFibers': True,
                        'simulateCells': simCell,
                        'simulateExtraEnvironment': True
                        }

        config['Scanning Parameters'] = {
                'Delta': 10,
                'dt': 0.001,
                'voxelDim': 50,
                'buffer': 10,
                'path_to_bvals': r'/bmr207/nmrgrp/nmr202/MCSIM/Repo/DBSI-99/bval-99.bval',
                'path_to_bvecs': r'/bmr207/nmrgrp/nmr202/MCSIM/Repo/DBSI-99/bvec-99.bvec'
                }

        config['Saving Parameters'] = {
                'path_to_save_file_dir': r'/bmr207/nmrgrp/nmr202/MCSIM/output9x9/R={}_C={}'.format(rows[iFile], cols[iFile])
        }
        config.write(cfgfile)
        cfgfile.close()
