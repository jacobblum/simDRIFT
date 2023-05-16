    
import configparser
from fractions import Fraction
import os 
import numpy as np

#Fractions = [(0.20, 0.20), (0.25, 0.25), (0.35, 0.35), (0.45, 0.45), (0.55, 0.55), (0.65, 0.65), (0.70, 0.70), (0.75, 0.75), (0.8, 0.8)]
Fractions = [(0.25, 0.25), (0.30, 0.30), (0.35, 0.35), (0.45, 0.45), (0.50, 0.50), (0.55, 0.55), (0.60, 0.60), (0.65, 0.65), (0.7, 0.7)]
#Fractions = [(0.20, 0.80), (0.25, 0.75), (0.35, 0.65), (0.45, 0.55), (0.5, 0.5), (0.6, 0.4), (0.67, 0.33), (0.70, 0.30), (0.85, 0.15)]
#CellFracs = [0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225]
#CellFracs = [0.05, 0.10, 0.15, 0.25, 0.35, 0.40, 0.45, 0.50, 0.55]
#Diffusions = [(1.5, 2.5)]
#Thetas = [(0,10), (0,20), (0,30), (0,40), (0,50), (0,60), (0,70), (0,80), (0,90)]

row = 8
col = 0

for frac in Fractions:
        config = configparser.ConfigParser()
        cfgfile = open(r"/bmr207/nmrgrp/nmr202/MCSIM/newMosaic" + os.sep + "{}{}_Config.ini".format(row, col), 'w')
        config['Simulation Parameters'] = {
                        'numSpins': 250*10**3,
                        'fiberFraction': frac,
                        'fiberRadius': 1.0,
                        'Thetas': (0, 45),
                        'fiberDiffusions': (1.5, 2.5),
                        'cellFraction': 0.20,
                        'cellRadii': (2.5, 5.0),
                        'fiberConfiguration': 'Interwoven',
                        'simulateFibers': True,
                        'simulateCells': True,
                        'simulateExtraEnvironment': True
                        }

        config['Scanning Parameters'] = {
                'Delta': 10,
                'dt': 0.001,
                'voxelDim': 50,
                'buffer': 5,
                'path_to_bvals': r"/bmr207/nmrgrp/nmr202/MCSIM/Repo/DBSI/bval-99.bval",
                'path_to_bvecs': r"/bmr207/nmrgrp/nmr202/MCSIM/Repo/DBSI/bvec-99.bvec"
                }

        config['Saving Parameters'] = {
                'path_to_save_file_dir': r"/bmr207/nmrgrp/nmr202/MCSIM/newMosaicSave/"
        }
        config.write(cfgfile)
        cfgfile.close()
        col+=1
