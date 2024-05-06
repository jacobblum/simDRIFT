import numpy as np 
import matplotlib.pyplot as plt 
import math
from tqdm import tqdm
from typing import Union, Type, Dict, List
import time 
from src.setup.objects import fiber, spin, cell
import os 
import logging

Npts   = 128 + 1
Ntheta = 64
def vec_2_frame(n : np.ndarray) -> np.ndarray:
    if np.ndim(n) < 3:
        n = n.reshape(1, n.shape[0], n.shape[1])

    n /= np.linalg.norm(n, ord = 2, axis = 2)[..., None]
    h   = np.stack([np.amax(np.stack([n[...,0] + 1, n[...,0] -1], axis = 1 ), axis = 1), 
                          n[...,1], 
                          n[...,2]
                          ], axis = 2
                          )
    hht = np.einsum('...i,...j -> ...ij', h, h)
    hth = np.einsum('bni, bni -> bn', h, h)
    H = np.eye(3) - 2 * hht / hth[..., None, None]
    tangent  = H[...,:,1]
    binormal = H[...,:,2] 
    return np.stack([n, tangent, binormal], axis = 3).squeeze()

def Ru(u : np.ndarray, theta : float) -> np.ndarray:
    if np.ndim(theta) < 1:
        theta = np.array([theta])
    if np.ndim(u) < 2:
        u = u.reshape(1, -1)
    u /= np.linalg.norm(u, ord = 2, axis=1)[:, None]
    Ru = np.stack([np.einsum('n, ij -> nij', np.cos(theta), np.eye(3)) + np.einsum('n, ij -> nij', np.sin(theta), np.cross(np.eye(3), u[ii, :])) + np.einsum('n, ij -> nij', (1.0 - np.cos(theta)), np.outer(u[ii, :], u[ii, :])) for ii in range(u.shape[0])], axis = 0)
    return Ru # Ru [NFrames, Nthetas, 3, 3]

class VoxelSurfaceMesh:
    def __init__(self, fibers_list: Type[fiber], cell_list: Type[cell], results_directory : str ) -> None:
        self.geom_dir           = os.path.join(results_directory, 'geometry')
        self._fibers_list       = fibers_list
        self._cell_list         = cell_list

        self._bundle_props_list = self._parse_fiber_properties()
        self._calculate_discretized_voxel_geometry()
        pass
    
    def _parse_fiber_properties(self) -> List[Dict[str, Union[Type[fiber], float, np.ndarray]]]:
        Bundle_Dict = {}
        for index, fiber in enumerate(self._fibers_list):
            if not (fiber.bundle in Bundle_Dict.keys()):
                Bundle_Dict[fiber.bundle] = [fiber]
            else:
                Bundle_Dict[fiber.bundle].append(fiber)

        Output_Args = []
        for k, v in Bundle_Dict.items():
            Bundle_Params = {}
            Bundle_Params['object'   ] = v[0]
            Bundle_Params['direction'] = v[0].direction
            Bundle_Params['radius'   ] = v[0].radius
            ctrs = np.empty((len(v), 3))

            for i, v_i in enumerate(v):
                ctrs[i, :] = v_i.center
                
            Bundle_Params['centers'] = ctrs
            Output_Args.append(Bundle_Params)
            
        return Output_Args
    
    def _calculate_discretized_voxel_geometry(self) -> List[Dict[str, np.ndarray]]:

        logging.info('------------------------------')
        logging.info(' Plotting Voxel Surface Mesh  ')
        logging.info('------------------------------')    
        
        if not os.path.exists(self.geom_dir): os.mkdir(self.geom_dir)

        VERTICIES  = []
        FACES      = []

        Output_Args = []
        ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')  
        colors = ['orchid', 'limegreen', 'deeppink']

        bdyXmin = []
        bdyXmax = []
        bdyYmin = []
        bdyYmax = []
        bdyZmin = []
        bdyZmax = []

        for bundle_index, bundle_params in enumerate(self._bundle_props_list):
            Bundle_VF_Dict = {}
            ctrs        = bundle_params['centers']
            L           = bundle_params['object'].L  
            radius      = bundle_params['radius']      
            theta       = bundle_params['object'].theta
            Gr0         = Ru(np.array([.0, 1., 0.]), theta = theta).squeeze()
            S           = np.linspace(-L, L, Npts)
            r           = np.stack([np.zeros(S.shape[0]), np.zeros(S.shape[0]), S], axis = 1)
            

            thetas          = np.linspace(0., 2*np.pi, Ntheta)
            fiber_trace     = np.zeros((r.shape[0], 3))
            surface_normals = np.zeros((r.shape[0], 3))

            for ii in range(r.shape[0]):
                fiber_trace[ii, :]     =  Gr0.dot(bundle_params['object']._gamma(Gr0.dot(r[ii, :])))
                surface_normals[ii, :] =  Gr0.dot(bundle_params['object']._d_gamma__d_t(Gr0.dot(r[ii, :])))
            
            fiber_traces = ctrs[:, None] + fiber_trace
            local_orthogonal_frames = vec_2_frame(surface_normals)
            Gr1                     = Ru(local_orthogonal_frames[:, :, 1], theta = np.pi / 2)
            surface_tangents        = np.einsum('FTij, Fj -> Fi', Gr1, surface_normals)
            Gr2                     = Ru(local_orthogonal_frames[:, :, 0], theta = thetas)    # Nframes, Nthetas, 3,3 
            surface_binormals       = np.einsum('FTij, Fj -> FTi', Gr2, surface_tangents) 
            surface_verticies       = fiber_traces[:, :, None] + (radius *surface_binormals)

            assert all([np.isclose(np.einsum('FTi, Fi -> F', surface_binormals,  surface_normals ), 0).all(),np.isclose(np.einsum('ni, ni -> n',  surface_tangents, surface_normals), 0).all() ]), "something's gone wrong. \
                The calculated surface tangent, normal, and binormal vectors are not orthogonal (or within 1e-08 of orthogonality). "

            surf_verticies_reshaped = surface_verticies.reshape(surface_verticies.shape[0], surface_verticies.shape[1]*surface_verticies.shape[2] , 3)
            
            
            bdyXmin.append(np.amin(surf_verticies_reshaped[:,:,0]))
            bdyXmax.append(np.amax(surf_verticies_reshaped[:,:,0]))
            bdyYmin.append(np.amin(surf_verticies_reshaped[:,:,1]))
            bdyYmax.append(np.amax(surf_verticies_reshaped[:,:,1]))
            bdyZmin.append(np.amin(surf_verticies_reshaped[:,:,2]))
            bdyZmax.append(np.amax(surf_verticies_reshaped[:,:,2]))
            
            N_Triangulations = 2
            for Nfiber in range(surface_verticies.shape[0]):
                Triangles = np.zeros((surface_verticies.shape[1]-1, 2*(surface_verticies.shape[2]-1), 3)) 
                Endcaps   = np.zeros((surface_verticies.shape[2] // 2 - 1, 4, 3))

                for ii in range(surface_verticies.shape[2] // 2 - 1):
                
                    seed_tris = (np.stack([
                                          ii                + np.array([0, surface_verticies.shape[1]*surface_verticies.shape[2] - surface_verticies.shape[2]]),
                                          (ii + 1)          + np.array([0, surface_verticies.shape[1]*surface_verticies.shape[2] - surface_verticies.shape[2]]),
                                          Ntheta - (ii + 1) + np.array([0, surface_verticies.shape[1]*surface_verticies.shape[2] - surface_verticies.shape[2]])],
                                           axis = 1
                                        )
                                )
                    
                    adj_tris = (np.stack([
                                         (ii + 1)           + np.array([0, surface_verticies.shape[1]*surface_verticies.shape[2] - surface_verticies.shape[2]]),
                                          Ntheta - (ii + 2) + np.array([0, surface_verticies.shape[1]*surface_verticies.shape[2] - surface_verticies.shape[2]]),
                                          Ntheta - (ii + 1) + np.array([0, surface_verticies.shape[1]*surface_verticies.shape[2] - surface_verticies.shape[2]])],
                                          axis = 1
                                          )
                                )
                    
                    Endcaps[ii, [0,2], :] = seed_tris
                    Endcaps[ii, [1,3], :] = adj_tris

                Endcaps_Linear = Endcaps.reshape(-1, 3)

                for ii in range(surface_verticies.shape[2] - 1):
                    pos_tris = np.stack([ii + np.arange(1,  surface_verticies.shape[1])*surface_verticies.shape[2] - surface_verticies.shape[2],
                                         ii + np.arange(1,  surface_verticies.shape[1])*surface_verticies.shape[2],
                                         ii + np.arange(1,  surface_verticies.shape[1])*surface_verticies.shape[2] - surface_verticies.shape[2] + 1
                                        ], axis = 1)
                    

          
                
                    neg_tris = np.stack([ii + np.arange(1, surface_verticies.shape[1])*surface_verticies.shape[2] - surface_verticies.shape[2] + 1,
                                        ii + np.arange(1, surface_verticies.shape[1])*surface_verticies.shape[2] + 1,
                                        ii + np.arange(1, surface_verticies.shape[1])*surface_verticies.shape[2]
                                        ], axis = 1)

                    Triangles[:, ii*N_Triangulations,    :] = pos_tris
                    Triangles[:, ii*N_Triangulations+1,  :] = neg_tris
                   
                Triangles_Linear = Triangles.reshape(-1, 3)
                Triangles_Linear = np.concatenate([Triangles_Linear, Endcaps_Linear], axis = 0)
            
                ax.plot_trisurf(surf_verticies_reshaped[Nfiber, :, 0].flatten(), 
                                surf_verticies_reshaped[Nfiber, :, 1].flatten(), 
                                surf_verticies_reshaped[Nfiber, :, 2].flatten(), 
                                triangles = Triangles_Linear,
                                color = colors[bundle_index],
                                lw=0.1, edgecolor="black", shade = False,
                                )
                    
                VERTICIES.append(surf_verticies_reshaped[Nfiber, :])
                FACES.append(Triangles_Linear)
        

        VERTICIES_npy, FACES_npy = np.concatenate(VERTICIES, axis = 0), np.concatenate(FACES, axis = 0)
        
        
        np.save(file = os.path.join(self.geom_dir, 'verticies.npy'), arr = VERTICIES_npy)
        np.save(file = os.path.join(self.geom_dir, 'faces.npy')    , arr = FACES_npy )        
        
        ### Plot Boundary ###

        buffer = 10.0e-6
        bdyYmin = min(bdyYmin) - buffer
        bdyYmax = max(bdyXmax) + buffer
        bdyXmin = min(bdyXmin) - buffer
        bdyXmax = max(bdyXmax) + buffer
        bdyZmin = min(bdyZmin) - buffer
        bdyZmax = max(bdyZmax) + buffer

        ## I did the triangulation manually, and started with the 1 index, so this array is zero-padded at the 0 index! 

        bdy_verticies = np.array([[0., 0., 0.],
                                  [bdyXmin, bdyYmin, bdyZmin],
                                  [bdyXmax, bdyYmin, bdyZmin],
                                  [bdyXmax, bdyYmax, bdyZmin],
                                  [bdyXmin, bdyYmax, bdyZmin],
                                  [bdyXmin, bdyYmin, bdyZmax],
                                  [bdyXmax, bdyYmin, bdyZmax],
                                  [bdyXmax, bdyYmax, bdyZmax],
                                  [bdyXmin, bdyYmax, bdyZmax]])
        
        triangles     = np.array([[1,2,3],
                                  [3,4,1],
                                  [1,2,6],
                                  [6,5,1],
                                  [2,7,6],
                                  [2,3,7],
                                  [7,4,3],
                                  [7,4,8],
                                  [5,8,4],
                                  [5,1,4],
                                  [6,5,7],
                                  [5,8,7]])

        ax.plot_trisurf(bdy_verticies[:, 0], 
                        bdy_verticies[:, 1], 
                        bdy_verticies[:, -1],
                        triangles = triangles, 
                        color = 'lightskyblue', 
                        alpha = .15,
                        shade = False)
        
        # Plot Spins
        Nspins = 500
        spin_positions_t1m = np.vstack([np.random.uniform(low=bdyXmin, high = bdyXmax, size=Nspins),
                                        np.random.uniform(low=bdyYmin, high = bdyYmax, size=Nspins),
                                        np.random.uniform(low=bdyZmin, high = bdyZmax, size=Nspins)])
        

        ax.scatter(spin_positions_t1m[0, :],
                   spin_positions_t1m[1, :], 
                   spin_positions_t1m[2, :], 
                   color = 'blue', 
                   s = 5, 
                   alpha = .20)


        def plt_sphere(list_center, list_radius):
            for c, r in zip(list_center, list_radius):
                # draw sphere
                u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
                x = r*np.cos(u)*np.sin(v)
                y = r*np.sin(u)*np.sin(v)
                z = r*np.cos(v)
                ax.plot_surface(x+c[0], y+c[1], z+c[2], color='orange', alpha=0.5)
        
        for cell in self._cell_list:
            plt_sphere([cell.center], [cell.radius])

        # Format Plot
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.view_init(elev = 25, azim = 45)

        # Format Ticks and Tick Labels
       
        ax.xaxis.set_ticks(ax.get_xticks())
        ax.xaxis.set_ticklabels( np.around(ax.get_xticks() * 1e6, 3))

        ax.yaxis.set_ticks(ax.get_yticks())
        ax.yaxis.set_ticklabels( np.around(ax.get_yticks() * 1e6, 3))

        ax.zaxis.set_ticks(ax.get_zticks())
        ax.zaxis.set_ticklabels( np.around(ax.get_zticks() * 1e6, 3))

        ax.set_xlabel(r'$x \; [\mu m]$')
        ax.set_ylabel(r'$y \; [\mu m]$')
        ax.set_zlabel(r'$z \; [\mu m]$')
        plt.savefig(os.path.join(self.geom_dir, 'meshed_voxel_geometry.png'))

        logging.info(' Plotting complete!')
        logging.info('------------------------------')    

        return Output_Args

    def _plot_with_spins(self, spins : List[Type[spin]]):
        return











