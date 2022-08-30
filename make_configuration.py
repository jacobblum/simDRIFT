    
import configparser
import os 

fractions = [(.80,.80), (.70,.70), (.60,.60), (.40, .40)]
Angle = [(0,0), (0,90), (0,60), (0,30)]



for fraction in fractions:
        for theta in Angle:
                config = configparser.ConfigParser()
                cfgfile = open(r"C:\MCSIM\dMRI-MCSIM-main\run_from_config_test\IW" + os.sep + "simulation_configuration_Theta={}_Fraction={}.ini".format(fraction,theta), 'w')
                config['Simulation Parameters'] = {
                                'numSpins': 100*10**3,
                                'fiberFraction': fraction,
                                'fiberRadius': 1.0,
                                'Thetas': theta,
                                'fiberDiffusions': (1.0,1.0),
                                'cellFraction': 0.0,
                                'cellRadii': (3,10),
                                'fiberConfiguration': 'Inter-Woven',
                                'simulateFibers': True,
                                'simulateCells': False,
                                'simulateExtraEnvironment': True
                                }

                config['Scanning Parameters'] = {
                        'Delta': 10,
                        'dt': 0.001,
                        'voxelDim': 20,
                        'buffer': 10,
                        'path_to_bvals': r'C:\MCSIM\Repo\simulation_data\DBSI\DBSI-99\bval',
                        'path_to_bvecs': r'C:\MCSIM\Repo\simulation_data\DBSI\DBSI-99\bvec'
                        }

                config['Saving Parameters'] = {
                        'path_to_save_file_dir': r"C:\MCSIM\dMRI-MCSIM-main\run_from_config_test"
                }
                config.write(cfgfile)
                cfgfile.close()
