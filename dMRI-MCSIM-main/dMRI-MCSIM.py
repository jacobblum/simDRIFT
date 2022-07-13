import numpy as np
import multiprocessing
from multiprocessing import Pool
import os 
import sys
import time 
from functools import partial
from tqdm import tqdm

## Functions as Modules

import whereSpin
import inCell
import walkInWater
import walkInFiber
import walkInCell
import simulate
import MCSIMplots  

def Generate_Fibers(fiber_number, fiber_radius, voxel_dims, Crossing):
    fiber_number = fiber_number 
    fiber_radius = fiber_radius 

    fiber_xycordinate = np.zeros((fiber_number**2,7))
    fiber_zcordiate = np.zeros((fiber_number**2,7))
    fibers1d = np.linspace(voxel_dims[0]+fiber_radius,voxel_dims[1]-fiber_radius,fiber_number)
    fibers2d = np.outer(fibers1d, np.ones(len(fibers1d)))
    fib_xs, fib_ys = fibers2d.T, fibers2d
    fiber_xycordinate[:,0] = fib_xs.flatten()
    fiber_xycordinate[:,1] = fib_ys.flatten()
    fiber_zcordiate[:,2] = np.ones(len(fib_xs.flatten()))
    fiber_radii = np.array([fiber_radius*1.10, fiber_radius*1.20, fiber_radius])
    fiber_diffusions = np.array([2.0, 2.0, 2.0])
   

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

def Place_Spins(num_spins, voxel_dims, fiber_xycordinate, cell_centers):
    spins = np.zeros((num_spins,3))

    fiber_spawn = True     
    
    if fiber_spawn:
        for j in range(spins.shape[0]):
            fiber_idx = np.random.randint(0,fiber_xycordinate.shape[0])
            fiber = fiber_xycordinate[fiber_idx,:]
            if fiber[3] == 0:
                spins[j,0:2] = (fiber[0:2])
                spins[j,2] = np.random.uniform(0,200)
            else:
                spins[j,1] = fiber[1]
                spins[j,2] = fiber[0]
                spins[j,0] = np.random.uniform(low = 0, high = 200)
        return spins
    
    cell_spawn = False
    
    if cell_spawn:
        for j in range(spins.shape[0]):
            cell_idx = np.random.randint(0,cell_centers.shape[0])
            cell = cell_centers[cell_idx,:]
            spins[j,:] = cell[0:3]
        
        return spins

    else:
        return np.random.uniform(low = 0, high = 200, size = (num_spins, 3))
    
 

def main():
    num_fibers = 1 # FF = 33.183 %
    num_spins = 100000
    dt = .0010 #ms
    voxel_dims = np.array([0,200])
    cell_radii = np.array([5.0, 20.0]) #um
    delta = 1
    t1_n = 2
    t1_p = t1_n + delta
    t2_n = 22
    t2_p = t2_n + delta
    TE = 28


    Crossing = False 

    spin_trajectory = np.zeros((num_spins,int(TE/dt),3))
    fiber_xycordinate = Generate_Fibers(num_fibers, 1.0, voxel_dims = voxel_dims, Crossing=Crossing)

    cell_centers_Q1 = Generate_Cells(2, np.array([0,100,0,100,0,200]), np.array([20]), 0)
    cell_centers_Q2 = Generate_Cells(8,np.array([100,200,0,100,0,200]), np.array([5]), 0) # 8 x 8 x 8
    cell_centers_Q3 = Generate_Cells(8, np.array([0,100,100,200, 0,200]), np.array([5]),0) 
    cell_centers_Q4 = Generate_Cells(2,np.array([100,200, 100,200,0,200]), np.array([20]), 0) # 2 x 2 x 2 
    cell_centers = np.vstack((cell_centers_Q1, cell_centers_Q2, cell_centers_Q3, cell_centers_Q4))
    spins = Place_Spins(num_spins, voxel_dims, fiber_xycordinate, cell_centers)



    #fiber_xycordinate = np.array([[1.0, 1.0, 0, 1, 1.0, 2.0, 1]])
    #cell_centers = np.array([[1.0,1.0,1.0,0.0]])
    #spins = np.array([[4.0, 0.0, 3.0]])
    spin_loc_key, spin_in_fiber_info, spin_in_cell_info = whereSpin.where_spin(fiber_xycordinate, cell_centers, spins, Crossing = Crossing)

    print('water: %s' %len(spin_loc_key[spin_loc_key == 0]))
    print('vfib: %s' %str(len(spin_loc_key[(spin_loc_key == 1) | (spin_loc_key == 4)])))
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
    spin_trajectory = spin_trajectory[:, 0:int(TE/dt), :]

    #For Signals
    t1_start, t1_end = int(t1_n/dt), int(t1_p/dt)+1
    t2_start, t2_end = int(t2_n/dt), int(t2_p/dt)+1

    spin_trajectory_1 = spin_trajectory[:,t1_start:t1_end,:]
    spin_trajectory_2 = spin_trajectory[:, t2_start:t2_end, :]

    print(spin_trajectory_1.shape)
    print(spin_trajectory_2.shape)



    np.save(r"/Users/jacobblum/Desktop/MCSIM-Jacob/CrossingFibers/Data/22_0713/taj1_3_fibers_50k_no_crossing_fixed_old.npy", spin_trajectory_1)
    np.save(r"/Users/jacobblum/Desktop/MCSIM-Jacob/CrossingFibers/Data/22_0713/traj2_3_fibers_50k_no_crossing_fixed_old.npy", spin_trajectory_2)
    np.save(r"/Users/jacobblum/Desktop/MCSIM-Jacob/CrossingFibers/Data/22_0713/spin_positions_3_fibers_50k_no_crossing_fixed_old.npy", spin_positions)
    
    #fig, ax = plt.subplots(figsize = (10,3))
    #MCSIMplots.plot(fiber_xycordinate, cell_centers, spins, spin_loc_key, spin_trajectory, voxel_dims)
    return

if __name__ == "__main__":
    main()

