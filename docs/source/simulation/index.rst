*********************
Simulation Module
*********************

.. _mainEx:

Inputs and Launchers
==============================================

Parameters Class
------------------
.. _Parameters:

.. autoclass:: simulation.Parameters
     :members: 

Simulation Class
-------------------

.. autoclass:: simulation.dmri_simulation
     :members: 

.. autofunction:: simulation.dmri_sim_wrapper

.. autofunction:: simulation.run

.. _setupFuncs:

Setup Functions
=============================================

.. automodule:: set_voxel_configuration
     :members: _set_num_fibers, _set_num_cells, _place_fiber_grid, _place_cells, _place_spins

.. automodule:: spin_init_positions
     :members: _find_spin_locations, _find_spin_locations_kernel

.. _Class Objects:

Class Objects
=============================================

.. _cellObj:

Cells
-------------------------------

.. autoclass:: objects.cell
     :members: __init__, _set_center, _set_diffusivity, _set_radius, _get_center, _get_diffusivity, _get_radius

.. _fiberObj:

Fibers
-------------------------------

.. autoclass:: objects.fiber
     :members: __init__, _get_center, _get_bundle, _get_direction, _get_diffusivity, _get_radius

.. _spinObj:

Spins
-------------------------------

.. autoclass:: objects.spin
     :members: __init__, _set_fiber_index, _set_fiber_bundle, _set_position_t2p, _set_cell_index, _set_water_index, _get_position_t1m, _get_position_t2p, _get_fiber_index, _get_bundle_index, _get_cell_index, _get_water_index

Diffusion Physics
=============================================

Wrapper
-------------------------------

.. automodule:: diffusion
     :members: _diffusion_context_manager, _simulate_diffusion, _caclulate_volumes
     :undoc-members:

Diffusion in Cells
-------------------------------

.. autofunction:: walk_in_cell._diffusion_in_cell

Diffusion in Fibers
-------------------------------

.. autofunction:: walk_in_fiber._diffusion_in_fiber

Diffusion in Water
-------------------------------

.. autofunction:: walk_in_water._diffusion_in_water

Save Outputs
=============================================

.. automodule:: save
     :members: _add_noise, _signal, _generate_signals_and_trajectories, _save_data

Miscellaneous Functions
=============================================

.. automodule:: linalg
     :members: Ry, affine_transformation, dL2

References
=============================================
     .. [1] Yang D. M., Huettner J. E., Bretthorst G. L., Neil J. J., Garbow J. R., Ackerman J. J. H., "Intracellular water preexchange lifetime in neurons and astrocytes." *Magn Reson Med.* **2018**; 79(3):1616-1627. `DOI: 10.1002/mrm.26781 <http://doi.org/10.1002/mrm.26781/>`_; `PMID: 28675497 <https://pubmed.ncbi.nlm.nih.gov/28675497/>`_; `PMCID: PMC5754269 <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5754269/>`_.
     .. [2] Garyfallidis E., Brett M., Amirbekian B., Rokem A., van der Walt S., Descoteaux M., Nimmo-Smith I., and Dipy Contributors (2014). "DIPY, a library for the analysis of diffusion MRI data." Frontiers in Neuroinformatics, vol.8, no.8.
     .. [3] Hall, M. G., and Alexander, D. C. (2009). "Convergence and parameter choice for monte-carlo simulations of diffusion MRI." *IEEE Trans. Med. Imaging* 28, 1354\ \U+2013\ 1364. `DOI: 10.1109/TMI.2009.2015756 <http://doi.org/10.1109/TMI.2009.2015756/>`_
