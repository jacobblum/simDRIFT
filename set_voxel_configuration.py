from cProfile import label
from pickletools import read_unicodestring1
import numpy as np 
import numba 
from numba import jit, cuda, int32, float32
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math
import jp
import matplotlib.pyplot as plt
import time 
import os  
from tqdm import tqdm
import nibabel as nb
import glob as glob 
import configparser
from ast import literal_eval
from multiprocessing import Process
import shutil
import walk_in_fiber
import walk_in_cell
import walk_in_extra_environ
import spin_init_positions
import sys
import diffusion

def _set_num_fibers(fiber_fractions, fiber_radius, voxel_dimensions, buffer):
    num_fibers = []
    for i in range(len(fiber_fractions)):
        num_fiber = int(np.sqrt((fiber_fractions[i] * (voxel_dimensions + buffer)**2)/(np.pi*fiber_radius**2)))
        num_fibers.append(num_fiber)    
    sys.stdout.write('\n{} fiber grid...'.format(num_fibers))
    return num_fibers

def _set_num_cells(cell_fraction, cell_radii, voxel_dimensions, buffer):
    num_cells = []
    for i in range(len(cell_radii)):
        num_cells.append(int((0.5*cell_fraction*((voxel_dimensions+buffer)**3)/((4.0/3.0)*np.pi*cell_radii[i]**3))))
    sys.stdout.write('\n{} cells...'.format(num_cells))
    return num_cells

def _place_fiber_grid(fiber_fractions, num_fibers, fiber_radius, fiber_diffusions, voxel_dimensions, buffer, void_distance, rotation_matrix, fiber_configuration):
    print('called')
    first_step_cordinates = []
    for i in range(len(fiber_fractions)):
        fiber_cordinates = np.zeros((num_fibers[i]**2,6))
        fiberYs, fiberXs = np.meshgrid(np.linspace(0+fiber_radius, voxel_dimensions+buffer-fiber_radius, num_fibers[i]),
                                       np.linspace(0+fiber_radius, voxel_dimensions+buffer-fiber_radius, num_fibers[i]))
        fiber_cordinates[:,0] = fiberXs.flatten()
        fiber_cordinates[:,1] = fiberYs.flatten()
        fiber_cordinates[:,3] = fiber_radius
        voxel_index = np.where((fiber_cordinates[:,1] >= i*0.5*(voxel_dimensions+buffer)) & (fiber_cordinates[:,1] < (i+1)*0.5*(voxel_dimensions+buffer)))[0]
        first_step_cordinates.append(fiber_cordinates[voxel_index,:])
    output_one_arg = np.vstack([first_step_cordinates[0], first_step_cordinates[1]])

    if fiber_configuration == 'Inter-Woven':
        Ys_mod2 = np.unique(output_one_arg[:,1])[::2]
        index = np.ind1d(output_one_arg[:,1], Ys_mod2)
        fiber_cordinates_pre_rotation = output_one_arg[index,0:3]
    
    if fiber_configuration == 'Penetrating' or fiber_configuration == 'Void':
        index = np.where(output_one_arg[:,1] < 0.5*(voxel_dimensions+buffer))[0]
        fiber_cordinates_pre_rotation = output_one_arg[index, 0:3]

    rotated_cordinates = (rotation_matrix.dot(fiber_cordinates_pre_rotation.T)).T
    
    if rotated_cordinates.shape[0] > 0:
        z_correction = np.amin(rotated_cordinates[:,2])
        rotated_fibers = rotated_cordinates
        rotated_fibers[:,2] = rotated_fibers[:,2] + np.abs(z_correction)
        output_one_arg[index, 0:3], output_one_arg[index,4], output_one_arg[index,5], output_one_arg[[i for i in range(output_one_arg.shape[0]) if i not in index],5] = rotated_fibers, 1.0, fiber_diffusions[0], fiber_diffusions[1]

    if fiber_configuration == 'Void':
        null_index = np.where((output_one_arg[:,1] > 0.5*(voxel_dimensions+buffer)-0.5*void_distance)
                              & (output_one_arg[:,1] < 0.5*(voxel_dimensions+buffer)+0.5*void_distance))
        output_one_arg[null_index] = 0
    final_output_arg = output_one_arg[~np.all(output_one_arg == 0, axis = 1)]
    sys.stdout.write('\nfiber grid placed...')
    sys.stdout.write('\n')
    return final_output_arg

def _place_cells(num_cells, cell_radii, fiber_configuration, voxel_dimentions, buffer, void_dist):
    cell_centers_total = []

    if fiber_configuration == 'Void':
        regions = np.array([[0, voxel_dimentions+buffer, 0.5*(voxel_dimentions+buffer)-0.5*void_dist, 0.5*(voxel_dimentions+buffer)+0.5*void_dist, 0, voxel_dimentions+buffer], 
                   [0, voxel_dimentions+buffer, 0.5*(voxel_dimentions+buffer)-0.5*void_dist, 0.5*(voxel_dimentions+buffer)+0.5*void_dist, 0, voxel_dimentions+buffer]])  
    else:
         regions = np.array([[0,voxel_dimentions+buffer,0,0.5*(voxel_dimentions+buffer), 0,voxel_dimentions+buffer],
                             [0,voxel_dimentions+buffer,0.5*(voxel_dimentions+buffer),voxel_dimentions+buffer,0,voxel_dimentions+buffer]])
    for i in (range(len(num_cells))):
            cellCenters = np.zeros((num_cells[i], 4))
            for j in range(cellCenters.shape[0]):
                sys.stdout.write('\r' + 'placed: ' + str((i+1)*(j+1)) + '/' + str(num_cells[0]+num_cells[1]) + ' cells')
                sys.stdout.flush()
                if j == 0:
                    invalid = True 
                    while(invalid):   
                        radius = cell_radii[i]
                        xllim, xulim = regions[i,0], regions[i,1]
                        yllim, yulim = regions[i,2], regions[i,3]
                        zllim, zulim = regions[i,4], regions[i,5]
                        cell_x = np.random.uniform(xllim + radius, xulim - radius)
                        cell_y = np.random.uniform(yllim + radius, yulim - radius)
                        cell_z = np.random.uniform(zllim + radius, zulim - radius)
                        cell_0 = np.array([cell_x, cell_y, cell_z, radius])
                        propostedCell = cell_0
                        ctr = 0
                        if i == 0:
                            cellCenters[j,:] = propostedCell
                            invalid = False
                        elif i > 0:
                            for k in range(cell_centers_total[0].shape[0]):
                                distance = np.linalg.norm(propostedCell-cell_centers_total[0][k,:], ord = 2)
                                if distance < (radius + cell_centers_total[0][k,3]):
                                        ctr += 1
                                        break
                        if ctr == 0:
                            cellCenters[j,:] = propostedCell
                            invalid = False
                elif (j > 0):
                    invalid = True
                    while(invalid):
                        xllim, xulim = regions[i,0], regions[i,1]
                        yllim, yulim = regions[i,2], regions[i,3]
                        zllim, zulim = regions[i,4], regions[i,5]
                        radius = cell_radii[i]
                        cell_x = np.random.uniform(xllim + radius, xulim - radius)
                        cell_y = np.random.uniform(yllim + radius, yulim - radius)
                        cell_z = np.random.uniform(zllim + radius, zulim - radius)
                        propostedCell = np.array([cell_x, cell_y, cell_z, radius])
                        ctr = 0
                        for k in range(j):
                            distance = np.linalg.norm(propostedCell-cellCenters[k,:], ord = 2)
                            if distance < 2*radius:
                                ctr += 1
                                break
                            if i > 0:
                                for l in range(cell_centers_total[0].shape[0]):
                                    distance = np.linalg.norm(propostedCell-cell_centers_total[0][l,:], ord = 2)
                                    if distance < (radius + cell_centers_total[0][l,3]):
                                        ctr += 1
                                        break
                        if ctr == 0:
                            cellCenters[j,:] = propostedCell
                            invalid = False
            cell_centers_total.append(cellCenters)
    output_arg = np.vstack([cell_centers_total[0], cell_centers_total[1]])
    sys.stdout.write('\n')
    return output_arg
        
def _generate_rot_mat(thetas):
    rotation_reference = np.zeros((len(thetas),3))
    for i, theta in enumerate(thetas):
        theta = np.radians(theta)
        s, c = np.sin(theta), np.cos(theta)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        rotation_matrix = Ry
        z = np.array([0,0,1])
        rotation_reference[i,:] = Ry.dot(z)
    sys.stdout.write('\nRotation Matrix: \n {}'.format(str(rotation_matrix)))
    return rotation_reference, rotation_matrix
