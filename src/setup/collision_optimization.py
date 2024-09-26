import numpy as np
import sys
import random
import logging
import src.setup.spin_init_positions as spin_init_positions
import src.setup.objects as objects
from src.jp import linalg
from src.setup.mesh import VoxelSurfaceMesh
from numba import jit, njit, cuda
from src.jp import linalg
import numba 
from collections import Counter
from itertools import product, zip_longest
from typing import Union, Type, Dict, List, Tuple
import time 
from src.setup.objects import fiber, spin, cell
import cupy as cp 
import math


import torch


def _package_data(self) -> Dict[str, Dict[str, Type[numba.cuda.cudadrv.devicearray.DeviceNDArray]]]:
    outputArgs = {'fiber_centers'     : {'data' : [], 'dtype' : np.float32},
                  'fiber_directions'  : {'data' : [], 'dtype' : np.float32},
                  'fiber_step'        : {'data' : [], 'dtype' : np.float32}, 
                  'fiber_radii'       : {'data' : [], 'dtype' : np.float32},
                  'fiber_theta'       : {'data' : [], 'dtype' : np.float32},
                  'curvature_params'  : {'data' : [], 'dtype' : np.float32},
                  'spin_positions_t0' : {'data' : [], 'dtype' : np.float32},
                  'spins_fiber_index' : {'data' : [], 'dtype' : np.int32  },
                  'cell_centers'      : {'data' : [], 'dtype' : np.float32},
                  'spins_cell_index'  : {'data' : [], 'dtype' : np.int32  },
                  'cell_step'         : {'data' : [], 'dtype' : np.float32},
                  'cell_radii'        : {'data' : [], 'dtype' : np.float32},
                  'spin_water_index'  : {'data' : [], 'dtype' : np.int32  },
                  'water_step'        : {'data' : [], 'dtype' : np.float32},
                  'gradient'          : {
                                        'data' : self.G, 
                                        'dtype': np.float32
                                        },
                  'phase'             : {
                                        'data' : np.zeros((len(self.spins), self.G.shape[0])), 
                                        'dtype': np.float32
                                        },

                 'nans'              : {
                                        'data'  : np.zeros(28),
                                        'dtype' : np.float32
                                        },
                
                }
    
    #####################################################################################
    #                                       Package Data                                #
    #####################################################################################
    for fiber in self.fibers:
        outputArgs['fiber_centers'   ]['data'].append(fiber.center)                           
        outputArgs['fiber_directions']['data'].append(fiber.direction)     
        outputArgs['fiber_step'      ]['data'].append(np.sqrt(6.0*fiber.diffusivity*self.dt)) 
        outputArgs['fiber_radii'     ]['data'].append(fiber.radius)                             
        outputArgs['fiber_theta'     ]['data'].append(fiber.theta)
        outputArgs['curvature_params']['data'].append(
                                                      [fiber.__dict__['kappa'], fiber.__dict__['L'], fiber.__dict__['A'], fiber.__dict__['P']]
                                                      )
    for cell in self.cells:
        outputArgs['cell_centers']['data'].append(cell.center)
        outputArgs['cell_radii'  ]['data'].append(cell.radius)
        outputArgs['cell_step'   ]['data'].append(np.sqrt(6.0 * cell.diffusivity*self.dt))

    for spin in self.spins:
        outputArgs['spin_positions_t0']['data'].append(spin.position_t1m)
        outputArgs['spins_fiber_index']['data'].append(-1 if spin._get_bundle_index() is None else spin._get_fiber_index())
        outputArgs['spins_cell_index' ]['data'].append(spin._get_cell_index())
        outputArgs['spin_water_index' ]['data'].append(1 if np.logical_and(spin._get_bundle_index() is None, spin._get_cell_index() == -1) else -1)

    outputArgs['water_step']['data'].append(math.sqrt(6*self.water_diffusivity*self.dt))

    
    #####################################################################################
    #                              Send Data to GPU                                     #
    #####################################################################################
    for k, v in outputArgs.items():
        outputArgs[k]['data'] = torch.from_numpy(np.array(v['data'], dtype = v['dtype'])).float().to('cuda')
    
    return outputArgs




def _gamma_t(r, v, theta, cp, force_broadcast : bool = False) -> torch.cuda.FloatTensor:
    
    # Native Call
    if (r.shape[0] == v.shape[0]) and (not force_broadcast):
        g_t = torch.zeros( (r.shape[0], 3), device = 'cuda')
        t = torch.einsum('Ni, Ni -> N', r, v) # Distance Along Principal Fiber Axis 

    # Broadcasted Call 
    elif (r.shape[0] != v.shape[0]) or (force_broadcast):
        g_t = torch.zeros( (r.shape[0], v.shape[0], 3), device = 'cuda')
        t = torch.einsum('Mi, Ni -> MN', r, v)

    x = cp[:, 2] * torch.sin(torch.pi * cp[:, 0] /((1/cp[:, 3])*cp[:, 1])*t) 
    z = t 

    g_t[...,  0] =  torch.cos(theta)*x  + torch.sin(theta)*z
    g_t[..., -1] = -torch.sin(theta)*x  + torch.cos(theta)*z

    return g_t 

def _d_gt__d_t(r, v, theta, cp, force_broadcast : bool = False) -> torch.cuda.FloatTensor:
    
    if (r.shape[0] == v.shape[0]) and (not force_broadcast):
        d_gt__d_t = torch.zeros( (r.shape[0], 3), device = 'cuda')
        t = torch.einsum('Ni, Ni -> N', r, v) # Distance Along Principal Fiber Axis 

    elif (r.shape[0] != v.shape[0]) or (force_broadcast):
        d_gt__d_t = torch.zeros( (r.shape[0], v.shape[0], 3), device = 'cuda')
        t = torch.einsum('Mi, Ni -> MN', r, v)

    x = cp[:, 2] * (torch.pi*cp[:, 0] /((1/cp[:, 3])*cp[:, 1])) * torch.cos(torch.pi*cp[:, 0] /((1/cp[:, 3])*cp[:, 1])*t)
    
    l_2_norm = torch.sqrt( torch.square(x) + 1)

    x /= l_2_norm
    z = 1/l_2_norm

    d_gt__d_t[...,  0] = torch.cos(theta)*x   + torch.sin(theta)*z
    d_gt__d_t[..., -1] = -torch.sin(theta)*x + torch.cos(theta)*z

    return d_gt__d_t

class cube:
    def __init__(self, center : np.ndarray, dx : float, dy : float, dz : float) -> None:
        
        x,y,z = 0,1,2
        
        self.center    = center
        self.dx        = dx
        self.dy        = dy
        self.dz        = dz
        self.verticies = np.array(
                                    [
                                    [self.center[x] + dx, self.center[y] + dy, self.center[z] + dz],    
                                    [self.center[x] + dx, self.center[y] + dy, self.center[z] - dz], 
                                    [self.center[x] + dx, self.center[y] - dy, self.center[z] + dz],
                                    [self.center[x] + dx, self.center[y] - dy, self.center[z] - dz],
                                    [self.center[x] - dx, self.center[y] - dy, self.center[z] + dz],
                                    [self.center[x] - dx, self.center[y] - dy, self.center[z] - dz],
                                    [self.center[x] - dx, self.center[y] + dy, self.center[z] + dz],
                                    [self.center[x] - dx, self.center[y] + dy, self.center[z] - dz],
                                    ]
                                    )
        self.faces    = np.array(
                                    [
                                    [0, 1, 2],
                                    [2, 3, 1],
                                    [1, 3, 5],
                                    [5, 7, 1],
                                    [5, 7, 6],
                                    [6, 4, 5],
                                    [6, 0, 1],
                                    [6, 7, 1],
                                    [4, 2, 3],
                                    [5, 4, 2],
                                    [4, 6, 0],
                                    [4, 2, 0]
                                    ]
                                    )
        self.normal = np.array(
                                [
                                    [ 2*dx, 0, 0],
                                    [ 0, 2*dy, 0],
                                    [ 0, 0, 2*dz]
                                ]
                                )
        self.normal /= np.linalg.norm(self.normal, ord = 2, axis = 1)
        pass 



def _MCIntegrate(self, N_particles : int = 2500000) -> np.ndarray:
    x,y,z = 0,1,2







    


    return






def ind2patch(cubes : List[Type[cube]]) -> Dict[int, List[int]]:
    x,y,z = 0,1,2
    map = {} # cube index -> patch of nbhd. cubes
    N_partitions = int (np.cbrt(len(cubes)))
    for linear_index, c in enumerate(cubes):

        subscript = np.unravel_index(linear_index, (N_partitions, N_partitions, N_partitions))

        adj_patches_substricpt = [
                                    p for p in product( 
                                                range(max(0, subscript[x] - 1), min(N_partitions, subscript[x] + 2) ),
                                                range(max(0, subscript[y] - 1), min(N_partitions, subscript[y] + 2) ),
                                                range(max(0, subscript[z] - 1), min(N_partitions, subscript[z] + 2) )
                                                )
                                    ]
        
        adj_patches_linear = np.array(
                                        [
                                        np.ravel_multi_index(s_i, (N_partitions, N_partitions, N_partitions)) for s_i in adj_patches_substricpt
                                        ]
                                        )

        map[linear_index] = adj_patches_linear     
    return map 

def ind2patch_geom(ind2patch_map : Dict[int, List[int]], all_fiber_intersecting_cubes : np.ndarray, all_fiber_interior_points : np.ndarray) -> Dict[int, List[int]]:
    map = {}
    for k, v in ind2patch_map.items():
        ith_cube_plus_patch_microstructure = []
        for ii in range(all_fiber_interior_points.shape[0]):
            if np.intersect1d(np.argwhere(all_fiber_intersecting_cubes[ii, :] > 0), v).shape[0] > 0:
                ith_cube_plus_patch_microstructure.append(ii)
        map[k] = ith_cube_plus_patch_microstructure

    return map

def bdy_2_ctr_patches(bdy_cubes, ctr_cubes) -> Dict[int, List[int]]:
    r"""
    Map a cube in the boundary cube to any adjacent cubes in the interior cube
    -------------------------------------------------------------------------
    Idea - The verticies of the adjacent interior cube will be tangent to the boundary face of the boundary cube.
    Thus, compute the distance from the boundary cube center to the verticies, and if this distance is less than
    dx_bdy + some small number, then the vertex is tangnet to the face of the cube. 
    """
    eps = 1e-7
  
    bdy_cube_centers    = np.stack([c.center for c in bdy_cubes], axis = 0)
    bdy_cube_normals    = np.stack([c.normal for c in bdy_cubes], axis = 0)
    interior_cube_verts = np.stack([c.verticies for c in ctr_cubes], axis = 0)
    bdy_cube_dx = bdy_cubes[0].dx

    bdy_cube_ctr_2_int_cube_verts_vec = bdy_cube_centers[:, np.newaxis, np.newaxis, :] - interior_cube_verts[np.newaxis, ...]
    
    bdy_cube_ctr_2_int_cube_verts_dist = np.abs(
                                                np.einsum('PCVi, PDi -> PCVD', bdy_cube_ctr_2_int_cube_verts_vec, bdy_cube_normals)
                                               ) 

    indicies_array = np.argwhere(
                                ((bdy_cube_ctr_2_int_cube_verts_dist < bdy_cube_dx + eps).all(axis = -1)).any(axis = -1) > 0 
                                ) # -> [:, 0] = bdy_cube_index | [:, 1] = int_cube_index
    
    bdy_cube_ind_2_adj_int_cube_ind = {}
    
    for bdy_cube_index in indicies_array[:, 0]:
        bdy_cube_ind_2_adj_int_cube_ind[bdy_cube_index] = indicies_array[indicies_array[:, 0] == bdy_cube_index, 1]
    
    return bdy_cube_ind_2_adj_int_cube_ind


def sparse_2_dense_cube2geom_map(ind2patch_geom_map):
    r"""
    Retun a Dict with k,v pairs s.t. the k, a cube index in [0... N_cubes], points to a tensor 
    of equal size along dim = -1 so that einsum may be performed on this collection of cubes 
    at once.  
    
    """

    idxs, tensors = [], []

    counts = np.unique( (ind2patch_geom_map > 0).sum(axis = -1) )

    for count in counts:
        tensors.append(
                    
                         ind2patch_geom_map[ ((ind2patch_geom_map > 0).sum(axis = -1) == count), 0:count]
                         
                      )
        
        idxs.append(
                    list(
                        np.argwhere( (ind2patch_geom_map > 0).sum(axis = -1) == count ).flatten()
                        )
                   )

    return idxs, tensors



def _calc_int_pts(self, bdy_cubes, int_cubes):
    x,y,z = 0,1,2

    dx = int_cubes[0].dx
    micro_structure_elements = np.concatenate([[fiber.center for fiber in self.fibers], [cell.center for cell in self.cells]])
    x_min, x_max = np.amin(micro_structure_elements[..., x]), np.amax(micro_structure_elements[..., x])
    y_min, y_max = np.amin(micro_structure_elements[..., y]), np.amax(micro_structure_elements[..., y]) 
    z_min, z_max = 0., 0. + (y_max-y_min)


    N_s = int(256e3)


    for ii in range(1):

        pts = np.stack([
                        np.random.uniform(low = x_min, high = x_max, size = (N_s,)),
                        np.random.uniform(low = y_min, high = y_max, size = (N_s,)),
                        np.random.uniform(low = z_min, high = z_max, size = (N_s,)),
                        ], axis = 1)


        pts_1 = np.load(r"C:\Users\Jacob\20240522_0925_simDRIFT_Results\debug\iter_0_pts.npy") 


        cube_centers = np.stack([c.center for c in int_cubes])
        cube_normals = np.stack([c.normal for c in int_cubes])

    
        ith_spin_to_jth_cube_vect = cube_centers[np.newaxis, :, :] - pts_1[:, np.newaxis, :]
        ith_spin_to_jth_cube_dist = np.abs(
                                        np.einsum('PCi, CDi -> DCP', ith_spin_to_jth_cube_vect, cube_normals)
                                        )
        

        print(ith_spin_to_jth_cube_dist.dtype)

    

    r"""
    ith_spin_to_jth_cube_vect = cube_centers[cp.newaxis, :, :] - spin_positions[:, cp.newaxis, :] 
    ith_spin_to_jth_cube_dist = cp.abs(
                                   cp.einsum('PCi, CDi -> PCD', ith_spin_to_jth_cube_vect, cube_normals)
                                  )
    


    t = cp.all(ith_spin_to_jth_cube_dist > 0)

    exit()
    
    test = cp.all( (ith_spin_to_jth_cube_dist < dx), axis = -1)   
    
    
    
    exit()
    ith_spin_resident_cube = cp.argwhere( 
                                         ((ith_cube_to_spin_dist < dx).all(axis = 2)) > 0
                                        )[:, 0]
    

    """


    return


def _run(self, N_partitions : int = 5, N_bdy : int = 1) -> Tuple[Union[np.ndarray, Dict[int, List[int]]]]:
    x,y,z = 0,1,2
    r"""
    Returns A Dictionary whose key is a cube index and value is a list of all fibers within 1 cube of the index
    """
    import matplotlib.pyplot as plt 

    ### Voxel Length must be evenly divisable by N_partitions ??? Yes -> And idk why this makes sense but it's an easy fix really
    ### Yeah - this is true, just need (voxel_dim + 2*buffer) % N_partitions = 0.

    stuck_spins = np.load(r'C:\Users\Jacob\Box\MCSIM_for_ISMRM\simDRIFT\examples\stuck_spin_steps.npy')

    #####################################################################################
    #                                Construct Sub-Voxels                               #
    #####################################################################################

    micro_structure_elements = np.concatenate([[fiber.center for fiber in self.fibers]])

    x_min, x_max = np.amin(micro_structure_elements[..., x]), np.amax(micro_structure_elements[..., x])
    y_min, y_max = np.amin(micro_structure_elements[..., y]), np.amax(micro_structure_elements[..., y]) 
    z_min, z_max = 0., 0. + (y_max-y_min)


    cube_x, dx = np.linspace(x_min, x_max, num = N_partitions, retstep = True, endpoint=True)
    cube_y, dy = np.linspace(y_min, y_max, num = N_partitions, retstep = True, endpoint=True)
    cube_z, dz = np.linspace(z_min, z_max, num = N_partitions, retstep = True, endpoint=True)

    dx /= 2
    dy /= 2
    dz /= 2



    #####################################################################################
    #                                       Place Cubes                                 #
    #####################################################################################
    cube_xs, cube_ys, cube_zs = np.meshgrid(cube_x, cube_y, cube_z)
    
    cubes = [cube(
                  center = np.array([cube_xs[subscript], cube_ys[subscript], cube_zs[subscript]]),
                  dx = dx,
                  dy = dy,
                  dz = dz                  
                  ) for subscript in np.ndindex(cube_xs.shape)
            ]
    
    #####################################################################################
    #                   Calculate the i-th Cube's interior Microstructure               #
    #####################################################################################

    cube_centers = np.stack([c.center for c in cubes])
    cube_normals = np.stack([c.normal for c in cubes])


    N_s = int(2.5e5)


    p = np.stack([
                    np.random.uniform(low = x_min, high = x_max, size = (N_s,)),
                    np.random.uniform(low = y_min, high = y_max, size = (N_s,)),
                    np.random.uniform(low = z_min, high = z_max, size = (N_s,)),
                    ], axis = 1)


    r_p = torch.from_numpy(p).float().to('cuda')
    

    simulation_data = _package_data(self)

    jth_fiber_center           = simulation_data['fiber_centers']['data']  
    jth_fiber_radii            = simulation_data['fiber_radii']['data'] 
    jth_fiber_direction        = simulation_data['fiber_directions']['data']
    jth_fiber_curvature_params = simulation_data['curvature_params']['data']
    jth_fiber_theta            = simulation_data['fiber_theta']['data']


    ith_spins_jth_fiber_dynamic_ctr = jth_fiber_center + _gamma_t(r_p, 
                                                                  jth_fiber_direction, 
                                                                  jth_fiber_theta, 
                                                                  jth_fiber_curvature_params, 
                                                                  force_broadcast=True
                                                                 )
    
    ith_spins_jth_fiber_dynamic_normal = _d_gt__d_t(r_p, 
                                                    jth_fiber_direction, 
                                                    jth_fiber_theta, 
                                                    jth_fiber_curvature_params, 
                                                    force_broadcast=True
                                                   )
        
    d_kth_iter_ith_spin_to_jth_fiber         = r_p[:, None, :] - ith_spins_jth_fiber_dynamic_ctr
    d_l_2_sqr_kth_iter_ith_spin_to_jth_fiber = torch.square(torch.linalg.norm(d_kth_iter_ith_spin_to_jth_fiber, axis = -1, ord = 2))

    p     = torch.einsum('NFi, NFi -> NF', 
                         d_kth_iter_ith_spin_to_jth_fiber, 
                         ith_spins_jth_fiber_dynamic_normal
                        )

    p_sqr = torch.square(p)
    proj_dist_ith_spin_to_jth_fiber_along_p = torch.sqrt(d_l_2_sqr_kth_iter_ith_spin_to_jth_fiber - p_sqr)

    
    is_inside = torch.argwhere(proj_dist_ith_spin_to_jth_fiber_along_p < jth_fiber_radii)


    int_spin_index, int_fiber_index = is_inside[:, 0], is_inside[:, -1]
    


    p = r_p[int_spin_index, :]

    p = p.to('cpu').numpy()
    int_fiber_index = int_fiber_index.to('cpu').numpy()


    p_to_c_dist = p[:, None, :] - cube_centers[None, :, :] 

    p_to_c_oriented_dist = np.abs(
                                  np.einsum('NCi, CDi -> NCD', p_to_c_dist, cube_normals)
                                  )
    
    p_to_c_key = (p_to_c_oriented_dist < ( dx + 1 * math.sqrt(6.0 * self.water_diffusivity * self.dt) ) ).all(axis = -1)


    p_to_c_key = np.argwhere(p_to_c_key)

    ith_spin_index, ith_cube_index = p_to_c_key[:, 0], p_to_c_key[:, -1]


    ddd = stuck_spins[:, None, :] - cube_centers[None, :, :]

    dddd = np.abs(
                np.einsum('NCi, CDi -> NCD', ddd, cube_normals)
                )
    
    ddddd = (dddd < dx + 1 * math.sqrt(6.0 * self.water_diffusivity * self.dt) ).all(axis = -1)

    dddd = np.argwhere(ddddd)[:, -1]

    


    

    # p_to_c_key (N_in_fiber, ) The jth cube index
    # int_fiber_index (N_in_fiber, ) The jth fiber index

    cube_int_geom_key = [None] * len(cubes)

    
    L_max = 0

    for ii in range(len(cubes)):
        L_m_1 = L_max
        
        cube_int_geom_key[ii] = np.unique(int_fiber_index[ith_spin_index[ith_cube_index == ii]])
        
        if cube_int_geom_key[ii].shape[0] > L_m_1: 
            L_max = cube_int_geom_key[ii].shape[0]
    


    ax = plt.figure().add_subplot(projection='3d')

    for c_index, c in enumerate(cubes):

        ax.plot_trisurf(c.verticies[:, x],
                        c.verticies[:, y],
                        c.verticies[:, z],
                        triangles = c.faces,
                        color = 'gray',
                        alpha = .10
                        )
        
        if c_index in np.unique(dddd):

            for f_index, f in enumerate(self.fibers):
                if f_index in cube_int_geom_key[c_index]:
                    ax.scatter(self.fibers[f_index].center[x], self.fibers[f_index].center[y], self.fibers[f_index].center[z], color = 'black', s = 5)
                else:
                    ax.scatter(
                       f.center[x], 
                       f.center[y], 
                       f.center[z], 
                       color = 'red', s = 1
                       )




    ax.view_init(90, 0)
    plt.savefig(r'C:\Users\Jacob\Box\MCSIM_for_ISMRM\simDRIFT\examples\dev_geom_nw.png')


    ind2patch_geom_map_np = np.stack(
                                    [np.pad(
                                            cube_int_geom_key[ii], 
                                            (0, L_max - cube_int_geom_key[ii].shape[0]),
                                            'constant',
                                            constant_values=(-1,)
                                            ) 
                                     for ii in range(len(cubes))
                                    ]
                                    )


    bool_arr = (ind2patch_geom_map_np > 0)
    
    b_arr_sums = bool_arr.sum(axis = -1)

    unique_counts = np.unique( b_arr_sums )

    tensors  = [None] * unique_counts.shape[0]
    indicies = [None] * unique_counts.shape[0] 

    for index, count in enumerate(unique_counts):



        tensors[index]  = ind2patch_geom_map_np[ b_arr_sums == count, 0:count]  
        indicies[index] = np.argwhere( b_arr_sums == count).flatten()
       

    return cube_centers, cube_normals, tensors, indicies, dx

def _bdy_cubes(self, N_stdev : int = 1) -> None:
    x,y,z = 0,1,2
    #####################################################################################
    #             Construct Sub-Voxels Centered Around The Geom. Containing Cube        #
    #####################################################################################

    micro_structure_elements = np.concatenate([[fiber.center for fiber in self.fibers], [cell.center for cell in self.cells]])
    
    x_min, x_max = np.amin(micro_structure_elements[..., x]), np.amax(micro_structure_elements[..., x])
    y_min, y_max = np.amin(micro_structure_elements[..., y]), np.amax(micro_structure_elements[..., y]) 
    z_min, z_max = 0., 0. + (y_max-y_min)


    

    # Center Cube

    x_ctr_cube_ctr = self.voxel_dimensions / 2
    y_ctr_cube_ctr = self.voxel_dimensions / 2
    z_ctr_cube_ctr = z_max / 2 

    _dx = 6/4 * (x_max - x_min) 
    _dy = 6/4 * (y_max - y_min) 
    _dz = 6/4 * (z_max - z_min) 

    # Branch out from Center Cube

    _cube_x = np.arange(start = x_ctr_cube_ctr - N_stdev * _dx, stop = x_ctr_cube_ctr + (N_stdev+1)*_dx, step = _dx)
    _cube_y = np.arange(start = y_ctr_cube_ctr - N_stdev * _dy, stop = y_ctr_cube_ctr + (N_stdev+1)*_dy, step = _dy)
    _cube_z = np.arange(start = z_ctr_cube_ctr - N_stdev * _dz, stop = z_ctr_cube_ctr + (N_stdev+1)*_dz, step = _dz)

    cube_xs, cube_ys, cube_zs = np.meshgrid(_cube_x, _cube_y, _cube_z)

    cubes = [cube(
                  center = np.array([cube_xs[subscript], cube_ys[subscript], cube_zs[subscript]]),
                  dx = _dx / 2,
                  dy = _dy / 2,
                  dz = _dz / 2                  
                  ) for subscript in np.ndindex(cube_xs.shape)
            ]

    return cubes 
