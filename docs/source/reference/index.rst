Reference
=========

simulate
~~~~~~~~~~~

Command Line Options
--------------------
Typically, ``simDRIFT`` is run from the command line using the following command:

.. code-block :: bash

  (simDRIFT) >simDRIFT simulate --configuration PATH_TO_CONFIGURATION.INI FILE

Inputs
--------------------
`simDRIFT` relies on a CONFIGURATION.INI file to set important simulation parameters. These parameters include:

**Simulation Parameters**

- ``n_walkers (int) -`` The number of spins in the ensemble. We recommend using 256000 for quick, small scale simulations, and 1000000 for larger simulations. 
- ``Delta (float) -`` The diffusion time (Δ), in milliseconds (ms), for the simulated PGSE experiment. 
- ``dt (float) -`` The time step parameter, in milliseconds (ms), for the random walk. Because we adopt the narrow pulse approximation, this is also equal to the gradient duration (δ) used in the simulated PGSE experiment
- ``voxel_dims (float) -`` The side-length, in micrometers (μm), of the isotropic imaging voxel. 
- ``buffer (float) -`` The isotropic region, in micrometers (μm), beyond the imaging voxel where miscrostructural elements will still be populated. This parameter is useful for enforcing periodic boundary conditions.
- ``void_distance (float) -`` The distance, in micrometers (μm), between adjacent fiber bundles. 
- ``bvals (str) -`` The absolute path to a `.bval` file. The bval file should be in the FSL format, listing the b-value for each volume in the diffusion scheme. If you instead wish to use one of the already implemented diffusion schemes specified by the ``diffusion_scheme`` parameter, please enter 'N/A' here. Importantly, the absolute path specified here **MUST be preceeded by the "r" prefix to denote the string as literal!**
- ``bvecs (str) -`` The absolute path to a `.bvec` file. The bvec file should be in the FSL format, listing the gradient direction, with one column per volume. Each column of the text file should have three rows, describing a unit vector with x,y,z components. If you instead wish to use one of the already implemented diffusion schemes specified by the ``diffusion_scheme`` parameter, please enter 'N/A' here. Importantly, the absolute path specified here  **MUST be preceeded by the "r" prefix to denote the string as literal!**
- ``diffusion_scheme (str) -`` We have already implemented three widely used diffusion schemes: The DBSI, ABCD, and NODDI schemes. To use one of these schemes, enter 'DBSI_99', 'ABCD_103', or 'NODDI_145' respectively. If paths are entered for the ``bvecs`` and ``bvals`` parameters, those inputs will take precedence over the input entered here, and the entry will be ignored.
- ``output_directory (str) -`` The absolute path to the parent directory the simulation results will be stored under. If 'N/A' is entered, then the simulation will store results under the current working directory.  **MUST be preceeded by the "r" prefix to denote the string as literal!**
- ``verbose (str) -`` To suppress logging outputs, change the verbose parameter to 'no'. 

**Fiber Parameters**

- ``fiber_fractions (float) -`` The volume of the imaging voxel occupied by a fiber bundle. To place multiple fiber bundles, enter each fiber fraction seperated by a comma. For example, a single fiber bundle occupying 10 percent of the imaging voxel would be entered as ``fiber_fractions=0.10``, whereas three bundles each occupying 10 percent of the imaging voxel would be entered as ``fiber_fractions=0.10,0.10,0.10``. Theoretically, there is no maximum number of fiber bundles allowed, however, we recomend not exceeding three or four.
- ``fiber_radii (float) -`` The radius, in micrometers (μm), of each fiber in the fiber bundle defined by the ``fiber_fractions`` parameter. The length of arguments entered here must correspond with the length entered for the ``fiber_fractions``, i.e., a single fiber bundle containing fibers of radius 1 μm would be entered as ``fiber_radii= 0.10``, whereas three bundles each containing fibers of radius 1 μm, 1.5 μm, 1.5 μm respectively would be entered as ``fiber_radii=1.0,1.5,1.5`` 
- ``thetas (float) -`` The angle, in degrees, of each fiber in the fiber bundle with respect to the y-axis of the 3 dimensional imaging voxel. The length of arguments entered here must also correspond with the length entered for the ``fiber_fractions``
- ``fiber_diffusions (float) -`` The intrinsic diffusivity, in  :math:`μm^{2}/ms`, of water within the constituent fibers of bundles specified by the the ``fiber_fractions`` parameter. 

**Cell Parameters**

- ``cell_fractions (float) -`` The volume of the imaging voxel occupied by a cell population. To place multiple cell populations, enter each cell fraction seperated by a comma. We recomend not exceeding a total cell fraction of .40, as sphere packing in three dimensions becomes very difficult for high densities, which will increase the simulation runtime quite dramatically.
- ``cell_radii (float) -`` The radius, in micrometers (μm), of each cell in the cell populations defined by the ``cell_fractions`` parameter. The length of arguments entered here must correspond with the length entered for the ``cell_fractions``

**Water Parameter**

- ``water_diffusivity (float) -`` The intrinsic diffusivity, in  :math:`μm^{2}/ms`, of the extra-fiber water. For simulations of in-vivo diffusion, we reccomend 3.0 :math:`μm^{2}/ms` and for ex-vivo simulations, :math:`2.0 μm^{2}/ms`.

Example configuration file
--------------------
Below, please find an example of the structure of the CONFIGURATION.INI file used to run ``simDRIFT``.

.. code-block:: bash
    
    [SIMULATION]
    n_walkers = 256000
    Delta = .1
    dt = .001
    voxel_dims = 10
    buffer = 0
    void_distance = 0
    bvals = "r'PATH_TO_BVAL_FILE.bval'"
    bvecs = "r'PATH_TO_BVEC_FILE'.bvec"
    diffusion_scheme = 'DBSI_99'
    output_directory = "r'PATH_TO_OUTPUT_DIRECTORY'"
    verbose = 'yes'

    [FIBERS]
    fiber_fractions = 0,0
    fiber_radii = 1.0,1.0
    thetas = 0,0
    fiber_diffusions = 1.0,2.0
    
    [CELLS]
    cell_fractions = .1
    cell_radii = 1.0
    
    [WATER]
    water_diffusivity = 3.0


Outputs
--------------------
Under the directory specified by the **output_directory** parameter, simDRIFT will create a directory titled ``DATE_TIME_simDRIFT_results``. Within this directory the tool will produce the following files and directories:

* ``trajectories`` : A directory under which .npy files corresponding to the by-compartment *(cells, fiber, water, etc...)* and total initial (*trajectories_t1m*) and final (*trajectories_t2p*)
  spin positions are stored. The trajectory files may be useful for generating signals using a different diffusion scheme than the one provided 
  by the ``diff_scheme`` argument post-hoc. 

* ``signals`` : A directory under which NIfTI files containing the by-compartment and total signals generated from ``simDRIFT`` are stored. 

* ``log`` : A text file that contains a detailed description of the input parameters and a record of the simulation's execution.

* ``input_configuration``: A copy of the input INI configuration file so that simulation input parameters may be referenced or simulations may be reproduced in the future. 


Editing the Configuration File Within a Python Script
--------------------
For the purposes of running batches of many number of simulations, an existing ``CONFIGURATION.INI`` file may easily be modified from within a Python script. Below, please 
find an example code snippet used to modify a ``CONFIGURATION.INI`` used in the ``test suite``:

.. code-block:: Python

    import configparser

    cfg_file = configparser.ConfigParser()
    cfg_file.read(PATH_TO_CONFIG.INI FILE)

    cfg_file['SIMULATION']['n_walkers'] = '256000'
    cfg_file['SIMULATION']['DELTA'] = '.001'
    cfg_file['SIMULATION']['dt'] = '.001'
    cfg_file['SIMULATION']['voxel_dims'] = '10'
    cfg_file['SIMULATION']['buffer'] = '0'
    cfg_file['SIMULATION']['void_distance'] = '0'
    cfg_file['SIMULATION']['bvals'] = "r'PATH_TO_BVAL_FILE.bval'"
    cfg_file['SIMULATION']['bvecs'] = "r'PATH_TO_BVEC_FILE'.bvec"
    cfg_file['SIMULATION']['diffusion_scheme'] = "'DBSI_99'"
    cfg_file['SIMULATION']['output_directory'] = "r'PATH_TO_OUTPUT_DIRECTORY'"
    cfg_file['SIMULATION']['verbose'] = "'no'"

    cfg_file['FIBERS']['fiber_fractions'] = '0,0'
    cfg_file['FIBERS']['fiber_radii']= '1.0,1.0'
    cfg_file['FIBERS']['thetas'] = '0,0'
    cfg_file['FIBERS']['fiber_diffusions'] = '1.0,2.0'
        
    cfg_file['CELLS']['cell_fractions'] = '0,0'
    cfg_file['CELLS']['cell_radii'] = '1.0,1.0'

    cfg_file['WATER']['water_diffusivity'] = '3.0'

    with open(PATH_TO_CONFIG.INI FILE), 'w') as configfile:
        cfg_file.write(configfile)

Creating a Configuration File Within a Python Script
--------------------
If you wish to create a ``CONFIGURATION.INI`` file from within a Python script, please use the following example code as a reference:

.. code-block:: Python

    import configparser

    Config = configparser.ConfigParser()
    cfg_file = open(PATH_TO_CONFIG.INI FILE, 'w')

    Config.add_section('SIMULATION')
    Config.set('SIMULATION','n_walkers','256000')
    Config.set('SIMULATION','DELTA','.001')
    Config.set('SIMULATION','dt','.001')
    Config.set('SIMULATION','voxel_dims','10')
    Config.set('SIMULATION','buffer','0')
    Config.set('SIMULATION','void_distance','0')
    Config.set('SIMULATION','bvals',  "r'PATH_TO_BVAL_FILE.bval'")
    Config.set('SIMULATION','bvecs',  "r'PATH_TO_BVAL_FILE.bvec'")
    Config.set('SIMULATION','diffusion_scheme',"'DBSI_99'")
    Config.set('SIMULATION','output_directory',"r'PATH_TO_OUTPUT_DIRECTORY'")
    Config.set('SIMULATION','verbose',"'no'")

    Config.add_section('FIBERS')
    Config.set('FIBERS','fiber_fractions','0,0')
    Config.set('FIBERS','fiber_radii','1.0,1.0')
    Config.set('FIBERS','thetas','0,0')
    Config.set('FIBERS','fiber_diffusions','1.0,2.0')
    
    Config.add_section('CELLS')
    Config.set('CELLS','cell_fractions','0,0')
    Config.set('CELLS','cell_radii','1.0,1.0')

    Config.add_section('WATER')
    Config.set('WATER','water_diffusivity','3.0')

    Config.write(cfg_file)





