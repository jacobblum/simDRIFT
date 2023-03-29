from ast import Del
import multiprocessing as mp
from multiprocessing.sharedctypes import Value
import numpy as np 
import numba 
from numba import jit, njit, cuda
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math
import jp as jp
import time
import walk_in_fiber
import walk_in_cell
import walk_in_extra_environ
import sys


def _simulate_diffusion(spin_positions_t1m, 
                        spin_in_fiber1_at_index,
                        spin_in_fiber2_at_index, 
                        spin_in_cell_at_index, 
                        fiber_centers, 
                        cell_centers, 
                        Delta, 
                        dt, 
                        fiber_configuration, 
                        rotation_reference, 
                        **kwargs):
    random_state = kwargs.pop('random_state', 42)
    rng_states_gpu = cuda.to_device(create_xoroshiro128p_states(spin_positions_t1m.shape[0], seed = random_state))
    spin_positions_t1m_cpy = spin_positions_t1m.copy()
    spin_positions_t1m_gpu = cuda.to_device(spin_positions_t1m.astype(np.float32))
    spin_in_fiber1_at_index_gpu = cuda.to_device(spin_in_fiber1_at_index.astype(np.float32))
    spin_in_fiber2_at_index_gpu = cuda.to_device(spin_in_fiber2_at_index.astype(np.float32))
    fiber_centers_gpu = cuda.to_device(fiber_centers)
    spin_in_cell_at_index_gpu = cuda.to_device(spin_in_cell_at_index.astype(np.float32))
    cell_centers_gpu = cuda.to_device(cell_centers)
    rotation_reference_gpu = cuda.to_device(rotation_reference.astype(np.float32))
    dt_gpu = cuda.to_device(np.array([dt]).astype(np.float32))
    N_iter = int(Delta/dt)
    Start = time.time()
    threads_per_block = 64
    blocks_per_grid = (spin_positions_t1m_cpy.shape[0] + (threads_per_block-1)) // threads_per_block
    for i in range(N_iter):
        sys.stdout.write('\r' + 'Step: ' +  str(i+1) + '/' + str(N_iter))
        sys.stdout.flush()
        _diffusion_context_manager[blocks_per_grid,threads_per_block](rng_states_gpu, spin_positions_t1m_gpu, spin_in_fiber1_at_index_gpu, spin_in_fiber2_at_index_gpu, fiber_centers_gpu, spin_in_cell_at_index_gpu, cell_centers_gpu, rotation_reference_gpu, dt_gpu, (fiber_configuration == 'Void'))
        cuda.synchronize()
    End = time.time()
    sys.stdout.write('\nSimulation elapsed in: {} seconds'.format(round((End-Start)),3))
    Start = time.time()
    sys.stdout.write('\nTransfering trajectory data to the host device...')
    spin_positions_t2p = spin_positions_t1m_gpu.copy_to_host()
    End = time.time()
    sys.stdout.write('\nTrajectory data transfered to host device in: {} sec \n'.format(round(End-Start),3)) 
    
    return spin_positions_t2p, spin_positions_t1m_cpy


@numba.cuda.jit(fastmath=True)
def _diffusion_context_manager(rng_states, spin_positions, spin_in_fiber1_key, spin_in_fiber2_key, fiber_centers, spin_in_cell_key, cell_centers, fiber_rotation_reference, dt, void):
    """
    Parameters:

    rng_states:
    spinPositions: (N_spins, 3) numpy array
    spin_in_fiber_key: (N_spins, ) numpy array, the spin_in_fiber[i] is the fiber index of the i-th spin. -1 if spin not in fiber.
    """
    gpu_index = cuda.grid(1)
    dt = numba.float32(dt[0])
    if gpu_index > spin_positions.shape[0]:
        return
    
    spin_in_fiber1_index = int(spin_in_fiber1_key[gpu_index])
    spin_in_fiber2_index = int(spin_in_fiber2_key[gpu_index])
    spin_in_cell_index  = int(spin_in_cell_key[gpu_index])

    spin_in_fiber1_boolean = (spin_in_fiber1_index > -1)
    spin_in_fiber2_boolean = (spin_in_fiber2_index > -1)
    spin_in_cell_boolean  = ((spin_in_cell_index > -1) & (spin_in_fiber1_index == -1) & (spin_in_fiber2_index == -1)) 

    if spin_in_fiber1_boolean:
        walk_in_fiber._diffusion_in_fiber(gpu_index, rng_states, spin_in_fiber1_index, spin_positions, fiber_centers, fiber_rotation_reference, dt)
    
    if spin_in_fiber2_boolean:
        walk_in_fiber._diffusion_in_fiber(gpu_index, rng_states, spin_in_fiber2_index, spin_positions, fiber_centers, fiber_rotation_reference, dt)

    if spin_in_cell_boolean:
        walk_in_cell._diffusion_in_cell(gpu_index, rng_states, spin_positions, spin_in_cell_index, cell_centers, fiber_centers, fiber_rotation_reference, dt, void)
    
    if (not(spin_in_cell_boolean)) & (not(spin_in_fiber1_boolean)) & (not(spin_in_fiber2_boolean)):
        walk_in_extra_environ._diffusion_in_extra_environment(gpu_index,rng_states,spin_positions,fiber_centers, cell_centers, fiber_rotation_reference, dt)
    return
