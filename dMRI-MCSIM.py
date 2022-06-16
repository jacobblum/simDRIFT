import numpy as np
import multiprocessing
from multiprocessing import Pool
import os 
import sys
import time 
from functools import partial

## Functions as Modules

import whereSpin
import inCell
import walkInWater
import walkInFiber
import walkInCell
import simulate 

def Generate_Fibers(fiber_number, fiber_radius, voxel_dims, Crossing):
    fiber_number = fiber_number 
    fiber_radius = fiber_radius 

    fiber_xycordinate = np.zeros((fiber_number**2,7))
    fiber_zcordiate = np.zeros((fiber_number**2,7))
    fiber_radius = 1

    fibers1d = np.linspace(voxel_dims[0]+fiber_radius,voxel_dims[1]-fiber_radius,fiber_number)
    fibers2d = np.outer(fibers1d, np.ones(len(fibers1d)))
    fib_xs, fib_ys = fibers2d.T, fibers2d
    fiber_xycordinate[:,0] = fib_xs.flatten()
    fiber_xycordinate[:,1] = fib_ys.flatten()
    fiber_zcordiate[:,2] = np.ones(len(fib_xs.flatten()))
    fiber_radii = np.array([fiber_radius*1.20, fiber_radius*1.10, fiber_radius])
    fiber_diffusions = np.array([1.0, 1.5, 2.0])
   

    fiber_xycordinate[(fiber_xycordinate[:,0] <= voxel_dims[1]/2) & (fiber_xycordinate[:,1] <= voxel_dims[1]/2), 4:7] = np.array([fiber_diffusions[0], fiber_radii[0], 1.0])
    fiber_xycordinate[(fiber_xycordinate[:,0] > voxel_dims[1]/2) & (fiber_xycordinate[:,1] <= voxel_dims[1]/2),4:7] = np.array([fiber_diffusions[1], fiber_radii[1], 2.0])
    fiber_xycordinate[voxel_dims[1]/2 < fiber_xycordinate[:,1], 4:7] = np.array([fiber_diffusions[2], fiber_radii[2], 3.0])

    if Crossing:
        fiber_xycordinate[(fiber_xycordinate[:,1] > 100),3] = 1.0 
    
    #outputArg = fiber_xycordinate[(fiber_xycordinate[:,0] >= 100) | (fiber_xycordinate[:,1] >= 100)]

    return fiber_xycordinate

def Generate_Cells(cell_number, voxel_dims, cell_radii, vspacing):
    num_cells = cell_number
    cell_centers = np.zeros((num_cells**2*(vspacing), 4))
    cell_radii = cell_radii
    cell_xs, cell_ys, cell_zs = np.mgrid[voxel_dims[0]+np.amin(cell_radii):voxel_dims[1]-np.amin(cell_radii):(num_cells * 1j), voxel_dims[2]+np.amin(cell_radii):voxel_dims[3]-np.amin(cell_radii):(num_cells * 1j), voxel_dims[4]+np.amin(cell_radii):voxel_dims[5]-np.amin(cell_radii):((vspacing) * 1j)]
    cell_centers[:,0] = cell_xs.flatten()
    cell_centers[:,1] = cell_ys.flatten()
    cell_centers[:,2] = cell_zs.flatten()
    cell_centers[:,3] =cell_radii[0] * np.ones(len(cell_xs.flatten()))
    return cell_centers

def Place_Spins(num_spins, voxel_dims):
    x_llim, x_ulim = voxel_dims[0], voxel_dims[1]
    return np.random.uniform(low = x_llim, high = x_ulim, size = (num_spins,3))


def main():
    num_fibers = 40
    num_cells = 4
    num_spins = 25000
    TE = 27 #ms
    dt = .005 #ms
    voxel_dims = np.array([0,200])
    cell_radii = np.array([5.0, 20.0]) #um
    t1_n = 2
    t1_p = 8
    t2_n = 20
    t2_p = 26

    spin_trajectory = np.zeros((num_spins,int(TE/dt),3))
    fiber_xycordinate = Generate_Fibers(num_fibers, 1.0, voxel_dims = voxel_dims, Crossing=True)    
    cell_centers_Q1 = Generate_Cells(2, np.array([0,100,0,100,0,200]), np.array([20]), 3)
    cell_centers_Q2 = Generate_Cells(9,np.array([100,200,0,100,0,200]), np.array([5.0]), 10)
    cell_centers_Q3 = Generate_Cells(9, np.array([0,100,100,200, 0,200]), np.array([5.0]),10)
    cell_centers_Q4 = Generate_Cells(2,np.array([100,200, 100,200,0,200]), np.array([20]),3)
    cell_centers = np.vstack((cell_centers_Q1, cell_centers_Q2, cell_centers_Q3, cell_centers_Q4))
    spins = Place_Spins(num_spins, voxel_dims)

    #fiber_xycordinate = np.array([[1.0, 1.0, 0, 1, 1.0, 2.0, 1]])
    #cell_centers = np.array([[1.0,1.0,1.0,0.0]])
    #spins = np.array([[4.0, 0.0, 3.0]])
    spin_loc_key, spin_in_fiber_info, spin_in_cell_info = whereSpin.where_spin(fiber_xycordinate, cell_centers, spins, Crossing = True)

    print('water: %s' %len(spin_loc_key[spin_loc_key == 0]))
    print('vfib: %s' %str(len(spin_loc_key[spin_loc_key == 1])))
    print('hfib: %s' %str(len(spin_loc_key[spin_loc_key == 2])))
    print('cell: %s' %str(len(spin_loc_key[spin_loc_key == 3])))
  
    # Parallelize Code
    cpu_count = int(.80 *np.floor(multiprocessing.cpu_count())+1)
    
    print('############## CONNECTED TO: %s WORKERS ##############' %str(cpu_count))

    data_tuple = [(i, spins, spin_loc_key, spin_in_cell_info, spin_in_fiber_info, fiber_xycordinate, cell_centers, TE, dt) for i in range(num_spins)]

    start = time.time()
    p = multiprocessing.Pool(cpu_count)
    arr = p.map(simulate.Parallel, data_tuple)
    end = time.time()
    t = end - start 
    print('Compute Time: %s secs' %t)

    spin_trajectory = np.array(arr)
    spin_positions = spin_trajectory[:, int(TE/dt), 2]

    #For Signals
    t1_start, t1_end = int(t1_n/dt), int(t1_p/dt)+1
    t2_start, t2_end = int(t2_n/dt), int(t2_p/dt)+1
    spin_trajectory_1 = spin_trajectory[:,t1_start:t1_end,:]
    spin_trajectory_2 = spin_trajectory[:, t2_start:t2_end, :]

    np.save(r"/bmr207/nmrgrp/nmr107/MC-SIM/simulation_data/from_117/220613/taj1_10%25k.npy", spin_trajectory_1)
    np.save(r"/bmr207/nmrgrp/nmr107/MC-SIM/simulation_data/from_117/220613/traj2_10%25k.npy", spin_trajectory_2)
    np.save(r"/bmr207/nmrgrp/nmr107/MC-SIM/simulation_data/from_117/220613/spin_positions_10%25k.npy", spin_positions)
    
    
    #fig, ax = plt.subplots(figsize = (10,3))
    #MCSIMplots.plot(fiber_xycordinate, cell_centers, spins, spin_loc_key, spin_trajectory, voxel_dims)
    return


def nnlsq_homogenous(bvals, signal):
    num_ds = 2
    model = np.ones((bvals.shape[0], num_ds))
    for i in range(num_ds):
        if i == 1:
            model[:,i] = -(1/1000*bvals)
    cs = nnls(model, np.log(signal))[0]
    
    print('Homogenous')
    print(cs)

def nnlsq_model(bvals, signal):
    num_params = 140
    D = np.linspace(0, 3.5, num_params)
    Model = np.zeros((bvals.shape[0], num_params))
    for i in range(num_params):
        Model[:,i] = np.exp(-D[i] * bvals * 1/1000)
    cs = nnls(Model, signal)[0]
    cs = np.array(cs)
    print('Multi-Tensor')
    print(1/np.sum(cs) * cs[cs > 0])
    print(D[cs > 0])

if __name__ == "__main__":
    main()



# 49204