import numba 
from numba import jit, cuda
from src.jp import random, linalg, curvature 


@numba.cuda.jit(nopython=True,parallel=True)
def _diffusion_in_fiber(i, 
                        random_states, 
                        fiber_center, 
                        fiber_radius, 
                        fiber_direction, 
                        fiber_step, 
                        theta,
                        spin_positions,
                        curvature_params):
    """Simulated Brownian motion of a spin confined to within in a fiber, implemented via random walk with rejection sampling for proposed steps beyond the fiber membrane. Note that this implementation assumes zero exchange between compartments and is therefore only physically-accurate for :math:`\Delta < {\\tau_{i}}` [1]_. 

    :param i: Absolute index of the current thread within the block grid
    :type i: int
    :param random_states: ``xoroshiro128p`` random states
    :type random_states: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param fiber_center: Coordinates of the center of specified fiber
    :type fiber_center: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param fiber_radius: Radius of the specified fiber type
    :type fiber_radius: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param fiber_direction: Orientation of the specified fiber type
    :type fiber_direction: numba.cuda.cudadrv.devicearray.DeviceNDArray
    :param fiber_step: Distance travelled by resident spins for each time step :math:`\dd{t}`
    :type fiber_step: float
    :param spin_positions: Array containing the updated spin positions
    :type spin_positions: numba.cuda.cudadrv.devicearray.DeviceNDArray

    **Shapes**
        :random_states: 
            (n_walkers,) where n_walkers is an input parameter denoting the number of spins in the ensemble
        :cell_center: 
            (3,)
        :fiber_centers: 
            (n_fibers x n_fibers, 3) where n_fibers is computed in a manner such that the fibers occupy the supplied fiber fraction of the imaging voxel
        :fiber_radii: 
            (n_fibers x n_fibers, ) where n_fibers is computed in a manner such that the fibers occupy the supplied fiber fraction of the imaging voxel
        :fiber_directions: 
            (n_fibers x n_fibers, 3) where n_fibers is computed in a manner such that the fibers occupy the supplied fiber fraction of the imaging voxel
        :spin_positions: 
            (n_walkers, 3) where n_walkers is an input parameter denoting the number of spins in the ensemble  
    """
    
    eps = numba.float32(1e-3)
    previous_position       = cuda.local.array(shape = 3, dtype = numba.float32)
    proposed_new_position   = cuda.local.array(shape = 3, dtype = numba.float32)
    dynamic_fiber_center    = cuda.local.array(shape = 3, dtype = numba.float32)
    dynamic_fiber_direction = cuda.local.array(shape = 3, dtype = numba.float32)

    distance = fiber_radius + eps

    while(distance > fiber_radius):
   
        proposed_new_position = random.random_on_S2(random_states, 
                                                     proposed_new_position, 
                                                     i)
        
        for k in range(proposed_new_position.shape[0]):
            previous_position[k]     = spin_positions[i, k]
            proposed_new_position[k] = previous_position[k] + (fiber_step * proposed_new_position[k])
    
        dynamic_fiber_center = linalg.gamma(proposed_new_position,
                                                      fiber_direction, 
                                                      theta,
                                                      dynamic_fiber_center,
                                                      curvature_params
                                                      )
        
        dynamic_fiber_direction = linalg.d_gamma__d_t(proposed_new_position,
                                                      fiber_direction, 
                                                      theta,
                                                      dynamic_fiber_direction,
                                                      curvature_params  
                                                      )
        
        for k in range(dynamic_fiber_center.shape[0]): 
            dynamic_fiber_center[k] = dynamic_fiber_center[k] + fiber_center[k]

        distance = linalg.dL2(proposed_new_position, 
                              dynamic_fiber_center, 
                              dynamic_fiber_direction, 
                              True
                              )
        
    for k in range(proposed_new_position.shape[0]): spin_positions[i,k] = proposed_new_position[k]
     

r"""
python "C:\Users\Jacob\Box\MCSIM_for_ISMRM\simDRIFT\master_cli.py" simulate --configuration C:\Users\Jacob\Box\0303_2024_Simulations\theta_0\rho_fiber_0.6\config.ini

"""



