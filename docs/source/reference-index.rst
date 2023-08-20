Reference
=========

simulate
-----------------

Command Line Options
~~~~~~~~~~~~~~~~~~~

.. code-block :: bash

  (simDRIFT) >simDRIFT simulate --configuration CONFIGURATION.INI FILE

Inputs
~~~~~~~~~~~~~~~~~
`simDRIFT` is run from the command line, and relies on a CONFIGURATION.INI file to set important simulation parameters. These parameters include:

**Named Parameters**

- **n_walkers (int) -** The number of spins in the ensemble. We recommend using 256000 for quick, small scale simulations, and 1000000 for larger simulations. 
- **Delta (float) -** The diffusion time (Δ), in milliseconds (ms), for the simulated PGSE experiment. 
- **dt (float) -** The time step parameter, in milliseconds (ms), for the random walk. Because we adopt the narrow pulse approximation, this is also equal to the gradient duration (δ) used in the simulated PGSE experiment
- **voxel_dims (float) -** The side-length, in micrometers (μm), of the isotropic imaging voxel. 
- **buffer (float) -** The isotropic region, in micrometers (μm), beyond the imaging voxel where miscrostructural elements will still be populated. This parameter is useful for enforcing periodic boundary conditions.
- **void_distance (float) -** The distance, in micrometers (μm), between adjacent fiber bundles. 
- **bvals (str) -** The absolute path to a `.bval` file. The bval file should be in the FSL format, listing the b-value for each volume in the diffusion scheme. If you instead wish to use one of the already implemented diffusion schemes specified by the **diffusion_scheme** parameter, please enter 'N/A' here.
- **bvecs (str) -** The absolute path to a `.bvec` file. The bvec file should be in the FSL format, listing the gradient direction, with one column per volume. Each column of the text file should have three rows, describing a unit vector with x,y,z components. If you instead wish to use one of the already implemented diffusion schemes specified by the **diffusion_scheme** parameter, please enter 'N/A' here.
- **diffusion_scheme (str) -** We have already implemented three widely used diffusion schemes: The DBSI, ABCD, and NODDI schemes. To use one of these schemes, enter 'DBSI_99', 'ABCD_103', or 'NODDI_145' respectively. If paths are entered for the **bvecs** and **bvals** parameters, those inputs will take precedence over the input entered here, and the entry will be ignored.
- **output_directory (str) -** The absolute path to the parent directory the simulation results will be stored under. If 'N/A' is entered, then the simulation will store results under the current working directory. 
- **verbose (str) -** To suppress logging outputs, change the verbose parameter to 'no'. 
- **fiber_fractions (float) -** The volume of the imaging voxel occupied by a fiber bundle. To place multiple fiber bundles, enter each fiber fraction seperated by a comma. For example, a single fiber bundle occupying 10 percent of the imaging voxel would be entered as **fiber_fractions** = 0.10, whereas three bundles each occupying 10 percent of the imaging voxel would be entered as **fiber_fractions** =0.10,0.10,0.10. Theoretically, there is no maximum number of fiber bundles allowed, however, we recomend not exceeding three or four.
- **fiber_radii (float) -** The radius, in micrometers (μm), of each fiber in the fiber bundle defined by the **fiber_fractions** parameter. The length of arguments entered here must correspond with the length entered for the **fiber_fractions**, i.e., a single fiber bundle containing fibers of radius 1 μm would be entered as **fiber_radii** = 0.10, whereas three bundles each containing fibers of radius 1 μm, 1.5 μm, 1.5 μm respectively would be entered as **fiber_radii** =1.0,1.5,1.5 
- **thetas (float) -** The angle, in degrees, of each fiber in the fiber bundle with respect to the y-axis of the 3 dimensional imaging voxel. The length of arguments entered here must also correspond with the length entered for the **fiber_fractions**
- **fiber_diffusions (float) -** The intrinsic diffusivity, in  :math:`μm^{2}/ms`, of water within the constituent fibers of bundles specified by the the **fiber_fractions** parameter. 
- **cell_fractions (float) -** The volume of the imaging voxel occupied by a cell population. To place multiple cell populations, enter each cell fraction seperated by a comma. We recomend not exceeding a total cell fraction of .40, as sphere packing in three dimensions becomes very difficult for high densities, which will increase the simulation runtime quite dramatically.
- **cell_radii (float) -** The radius, in micrometers (μm), of each cell in the cell populations defined by the **cell_fractions** parameter. The length of arguments entered here must correspond with the length entered for the **cell_fractions**
- **water_diffusivity (float) -** The intrinsic diffusivity, in  :math:`μm^{2}/ms`, of the extra-fiber water. For simulations of in-vivo diffusion, we reccomend 3.0 μm^{2}/ms and for ex-vivo simulations, 2.0 μm^{2}/ms.

Example configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~
Below, please find an example of the structure of the CONFIGURATION.INI file used to run ``simDRIFT``.

.. code-block:: bash
    
    [SIMULATION]
    n_walkers = 256000
    Delta = .1
    dt = .001
    voxel_dims = 10
    buffer = 0
    void_distance = 0
    bvals = 'N/A'
    bvecs = 'N/A'
    diffusion_scheme = 'DBSI_99'
    output_directory = 'N/A'
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
~~~~~~~~~~~~~~~~~
Under the directory specified by the **output_directory** parameter, simDRIFT will create a directory titled ``DATE_TIME_simDRIFT_results``. Within this directory the tool will produce the following files and directories:

* ``trajectories`` : A directory under which .npy files corresponding to the by-compartment *(cells, fiber, water, etc...)* and total initial (*trajectories_t1m*) and final (*trajectories_t2p*)
  spin positions are stored. The trajectory files may be useful for generating signals using a different diffusion scheme than the one provided 
  by the ``diff_scheme`` argument post-hoc. 

* ``signals`` : A directory under which NIfTI files containing the by-compartment and total signals generated from ``simDRIFT`` are stored. 

* ``log`` : A text file that contains a detailed description of the input parameters and a record of the simulation's execution.

* ``input_configuration``: A copy of the input INI configuration file so that simulation input parameters may be referenced or simulations may be reproduced. 
