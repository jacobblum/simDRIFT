    
import configparser
    
config = configparser.ConfigParser()
cfgfile = open("simulation_configuration.ini", 'w')
config['Simulation Parameters'] = {
                  'numSpins': '100*10**3',
                  'fiberFraction': '(.82, .82)',
                  'fiberRadius': '1.0',
                  'Thetas': '(0,0)',
                  'fiberDiffusions': '(1.0,1.0)',
                  'cellFraction': '0.0',
                  'cellRadii': '(3,10)',
                  'penetrating': 'P',
                  'simulateFibers': 'True',
                  'simulateCells': 'False',
                  'simulateExtraEnvironment': 'True'
                  }

config['Scanning Parameters'] = {
        'Delta': '10',
        'dt': '.001',
        'voxelDim': '20',
        'buffer': '10',
        'path_to_bvals': r'C:\MCSIM\Repo\simulation_data\DBSI\DBSI-99\bval',
        'path_to_bvecs':  r'C:\MCSIM\Repo\simulation_data\DBSI\DBSI-99\bval'
        }

config['Saving Parameters'] = {
        'path_to_save_file_dir': 'rC:\MCSIM\dMRI-MCSIM-main\gpu_data'
}
config.write(cfgfile)
cfgfile.close()
