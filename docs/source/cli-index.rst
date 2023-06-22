************************
Command Line Interface
************************

Top-Level CLI (Module Selection)
==================================

.. autofunction:: master_cli.main

    Arguments
        1. ``simulate``: Used to run the simulator normally. See `Simulation CLI`_ below.
        2. ``run_tests``: Used to run the installation and accuracy tests. See `Testing CLI`_ below.

.. _Simulation CLI:

Simulation CLI
==================================

.. autoclass:: src.cli.CLI
    :members:

    Simulation Arguments
        1. ``n_walkers``: Number of spins to populate voxel. Must be an integer greater than zero. It is recommended to use a sufficient number of spins to achieve a spin density of at least 1.0 spin per cubic micrometer.
        2. ``fiber_fractions``: Volume fractions (as a tuple of floats) of each fiber type to populate in simulated voxel. Must be non-negative and sum to less than 1. 
        3. ``fiber_radii``: Radii of each desired fiber type (as a tuple of floats) in units of micrometers. Length must match the number of entries in ``fiber_fractions`` and value must be non-negative.
        4. ``thetas``: Angles for each fiber bundle to be rotated (w.r.t. the `y`-axis)
        5. ``fiber_diffusions``: Intrinsic diffusivities (in units of square micrometers per millisecond) of each fiber type. Length must match the number of entries in ``fiber_fractions`` and value must be non-negative.
        6. ``cell_fractions``: Volume fractions (as a tuple of floats) of each cell type to populate in simulated voxel. Must be non-negative and sum to less than 1. 
        7. ``cell_radii``: Radii of each desired cell type (as a tuple of floats) in units of micrometers. Length must match the number of entries in ``cell_fractions`` and be non-negative.
        8. ``water_diffusivity``: Diffusivity of free water (in units of square micrometers per millisecond). Must be non-negative.
        9. ``voxel_dims``: Side length for the simulated isotropic voxel, in units of micrometers. Must be a positive float.
        10. ``void_dist``: Size of edema-like gap to introduce between the fiber bundles.
        11. ``verbose``: ``yes`` or ``no``. Select ``yes`` to output detailed information to the terminal. Select ``no`` to suppress this output.
        12. ``Delta``: Diffusion time (or simulation length) in units of milliseconds.
        13. ``dt``: Size of time step for simulation.
        

.. _Testing CLI:

Testing CLI
==================================

.. autoclass:: tests.cli.CLI
    :members:

    Arguments
        None

    