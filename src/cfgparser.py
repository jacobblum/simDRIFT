import configparser
import os
from pathlib import Path
from src import gradients
import numpy as np 
from typing import Dict, Union

def parse_config_file(args : str) -> Dict[str, Union[float, np.ndarray, str]]:

    parsed_config_args = {}
    
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(Path(args).resolve())

    # N_walkers:  
    n_walkers = int(config['SIMULATION']['n_walkers'])
    assert n_walkers > 0, \
        f"The simulation must be performed with a non-negative number of spins"
    parsed_config_args['n_walkers'] = n_walkers

    # Delta (Diffusion Time [sec.])
    Delta = np.array(config['SIMULATION']['Delta'], dtype = np.float32)*1.0e-3 
    assert Delta > 0., \
        f"The simulation requires non-negative diffusion time."
    parsed_config_args['Delta'] = Delta

    # delta (Gradient Pulse Duration [sec.])
    delta = np.array(config['SIMULATION']['delta'], dtype=np.float32)*1.0e-3
    assert ((delta > 0)
            and delta < Delta), \
        f"delta must be non-negative AND < Delta"
    parsed_config_args['delta'] = delta

    parsed_config_args['TE'] = Delta + delta

    # dt (Timestep Duration [sec.])
    dt = np.array(config['SIMULATION']['dt'], dtype = np.float32)*1.0e-3
    assert ((dt > 0)
            and dt < Delta), \
        f"dt must be non negative AND < delta"
    parsed_config_args['dt'] = dt

    # voxel_dims (The Length of the Isotropic Image Voxel [meters])
    voxel_dimensions = float(config['SIMULATION']['voxel_dims'])*1.0e-6
    assert voxel_dimensions > 0., \
        f"The voxel must have positive length"
    parsed_config_args['voxel_dimensions'] = voxel_dimensions

    # void distance (The Distance between fiber bundles [meters])
    void_distance = float(config['SIMULATION']['void_distance'])*1.0e-6
    assert void_distance >= 0., \
        f"The void-distance must be non-negative"
    parsed_config_args['void_distance'] = void_distance

    # buffer (Additional area over which the voxel's microstructure is placed [meters] 
    # used for periodic boundary conditions
    buffer = float(config['SIMULATION']['buffer'])
    assert buffer >= 0., \
        f"The buffer must be non-negative"
    parsed_config_args['buffer'] = buffer

    # Custom bvals and bvecs files:
    USE_DEFAULT_DSCHEME = False
    for file in ['bvals', 'bvecs']:
        p = config['SIMULATION'][file].strip('\"').strip("\'")

        # Check that bvals or bvecs are not set to N/A, indicating useage of a built in diffusion scheme
        if (p == "N/A"):
            USE_DEFAULT_DSCHEME = True
            break
        
        # Cheak that the bval/bvec path exists and is read accessible
        assert os.access(p, os.R_OK), \
            f"Cannot read specified ({file[:-1]}) file: {p}.\n" \
            f"Ensure that the supplied file exists AND is read accessible."

        parsed_config_args[file] = p
    
    # Default Diffusion Scheme
    dscheme = "N/A"
    if USE_DEFAULT_DSCHEME:
        dscheme = config['SIMULATION']['diffusion_scheme'].strip("\'").strip('\"')
        assert dscheme in gradients.DEFAULT_DIFFUSION_SCHEME_LIST, \
            f"{dscheme} is not defined as a default diffusion scheme.\n" \
            f"please chose from any of {gradients.DEFAULT_DIFFUSION_SCHEME_LIST}." 
    
    parsed_config_args['dscheme'] = dscheme
    parsed_config_args['custom_diff_scheme_flag'] = USE_DEFAULT_DSCHEME

    # Output Directory (Where results will be stored)
    d = config['SIMULATION']['output_directory'].strip("\'").strip('\"')
    d = {'N/A' : None}[d] or os.getcwd()
    assert os.access(d, os.W_OK), \
        f"Ensure that the supplied output directory exists AND is write accessible."
    parsed_config_args['output_directory'] = d
    
    # Fiber Fractions (The volume of the image voxel occupied by the fibers)
    f_frac = np.array(
                     config['FIBERS']['fiber_fractions'].split(','),
                     dtype = np.float32
                     )
    assert ( (f_frac >= 0).all() 
            and (f_frac.sum() < 1.0) ), \
        f"fiber-fractions must be non-negative and sum to less than 1.0"
    parsed_config_args['fiber_fractions'] = f_frac
    
    # Fiber Radii (The radius of each fiber [meter])
    f_radii = np.array(
                     config['FIBERS']['fiber_radii'].split(','),
                     dtype = np.float32
                     ) * 1.0e-6
    assert (f_radii > 0).all(), \
        f"fiber-radii must be positive"
    parsed_config_args['fiber_radii'] = f_radii

    # Fiber Thetas
    f_theta = np.array(
                     config['FIBERS']['thetas'].split(','),
                     dtype = np.float32
                     ) 
    parsed_config_args['thetas'] = f_theta

    # Fiber Diffusions (The Intrinsic Diffusivity of intra-fiber water [m^2 / sec.])
    f_D0 = np.array(
                     config['FIBERS']['fiber_diffusions'].split(','),
                     dtype = np.float32
                     ) * 1.0e-9
    assert (f_D0 > 0).all(), \
        f"fiber-diffusivity must be non-negative"
    parsed_config_args['fiber_diffusions'] = f_D0

    # Kappa, the curvature of the fibers
    kappas = np.array(
                     config['CURVATURE']['kappa'].split(','),
                     dtype = np.float32
                     )
    parsed_config_args['kappa'] = kappas

    # Attenuation, how much the fibers curve [meters]
    A = np.array(
                config['CURVATURE']['Amplitude'].split(','),
                dtype = np.float32
                ) * 1.0e-6
    parsed_config_args['Amplitude'] = A

    # Periodicity, how many periods the fibers curve over the voxel_dimension
    P = np.array(
                config['CURVATURE']['Periodicity'].split(','),
                dtype = np.float32
                )
    assert (P >= 0).all(), \
        f"Periodicity must be greater than  or equal to 0"
    parsed_config_args['Periodicity'] = P

    # Assert that all parameters relating to fibers are self consistent
    assert len(set(
                [
                parsed_config_args[fp].shape[0] 
                for fp in list(config['FIBERS'].keys())[:-1] + list(config['CURVATURE'].keys())
                ]
                )) == 1, \
        f"Please ensure that all input FIBER and CURVATURE parameters\n " \
        f"Are self consistent (of equal length)."

    # Cell Fractions (The volume of the image voxel occupied by the cells)
    c_frac = np.array(
                     config['CELLS']['cell_fractions'].split(','),
                     dtype = np.float32
                     )  
    assert ((c_frac >= 0).all() 
            and (c_frac.sum() < 1.0)), \
        f"cell-fractions must be non-negative and sum to less than 1.0"
    parsed_config_args['cell_fractions'] = c_frac

    # Cell Radii (The Radius of the cells [meters])
    c_radii = np.array(
                     config['CELLS']['cell_radii'].split(','),
                     dtype = np.float32
                     ) * 1.0e-6
    assert (c_radii > 0).all(), \
        f"cell-radii must be positive"
    parsed_config_args['cell_radii'] = c_radii

    assert len(set(
                [
                parsed_config_args[cp].shape[0]
                for cp in list(config['CELLS'].keys())    
                ]
                )) == 1, \
        f"Please ensure that all input CELL parameters are\n"\
        f"self consistent (of equal length)"

    # Water Diffusivity (The Diffusivity of the water [m^2/s])

    w_D0 = np.array(
                     config['Water']['water_diffusivity'].split(','),
                     dtype = np.float32
                     ) * 1.0e-9 
    assert (w_D0 > 0).all(), \
        f"water-diffusivity must be non-negative"
    parsed_config_args['D0'] = np.sqrt(w_D0*6.0*parsed_config_args['dt'])

    return parsed_config_args