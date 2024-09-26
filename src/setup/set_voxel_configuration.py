import numpy as np
import sys
import logging
from scipy.spatial.transform import Rotation
from typing import Union, Type, Dict, List, Tuple
from src.setup.objects import fiber, spin, cell



logger = logging.getLogger('simDRIFT')

x,y,z = 0,1,2

class VoxelGeometry:
    def __init__(self,
                 n_walkers        : int ,
                 fiber_fractions  : np.ndarray,
                 fiber_radii      : np.ndarray, 
                 fiber_thetas     : np.ndarray,
                 fiber_diffusions : np.ndarray,
                 kappa            : np.ndarray,
                 amplitude        : np.ndarray,
                 periodicity      : np.ndarray,
                 cell_fractions   : np.ndarray,
                 cell_radii       : np.ndarray,
                 voxel_dimensions : float,
                 buffer           : float,
                 void_distance    : float,
                 dt               : float,
                 fiber_configuration : str = 'Interwoven'
                 ) -> None:
        
        self.n_walkers = n_walkers
        self.fiber_fractions = fiber_fractions
        self.fiber_radii = fiber_radii
        self.thetas = fiber_thetas
        self.fiber_diffusions = fiber_diffusions
        self.kappa = kappa
        self.Amplitude = amplitude
        self.Periodicity = periodicity
        self.cell_fractions = cell_fractions
        self.cell_radii = cell_radii
        self.voxel_dimensions = voxel_dimensions
        self.buffer = buffer 
        self.void_distance = void_distance
        self.dt = dt
        self.fiber_configuration = fiber_configuration


        # Calculate The Number of Fibers 
        self.n_fibers = self._calc_n_fibers()
        self.n_cells = self._calc_n_cells()

        # Place the fiber grid
        self.fibers = self._place_fiber_grid()

        # Place the cell grid
        self.cells = self._place_cell_lattice()

        # Place the spins in the image voxel
        self.spins = self._place_spins()

        pass

    def _calc_n_fibers(self) -> List[Type[fiber]]:
       
        logger.info('------------------------------')
        logger.info(' Fiber Setup')
        logger.info('------------------------------') 

        n_fibers = np.zeros(self.fiber_fractions.shape[0], dtype=int)
        
        bundle_idx = 0
        for f_i_frac, f_i_rad in zip(self.fiber_fractions, self.fiber_radii):

            vl = (self.voxel_dimensions + self.buffer) ** 2
            n_fiber_i = int(
                            np.sqrt( 
                                    n_fibers.shape[0] * ( vl * f_i_frac)/(np.pi*f_i_rad**2)
                                    )
                            )
              
            n_fibers[bundle_idx] = n_fiber_i
            bundle_idx += 1
            logger.info(f"{n_fiber_i**2 // 1} fibers of radius {round(1e6 * f_i_rad,3)} [um] in bundle {bundle_idx} will be placed in the voxel")
        
        return n_fibers

    def _calc_n_cells(self):
        logger.info('------------------------------')
        logger.info(' Cells Setup')
        logger.info('------------------------------')    
        
        n_cells = np.zeros(self.cell_fractions.shape[0], dtype=int)
        bundle_idx = 0
        for c_i_frac, c_i_rad in zip(self.cell_fractions, self.cell_radii):
            n_cell_i = int(
                          (c_i_frac*(self.voxel_dimensions**3)/((4.0/3.0)*np.pi*c_i_rad**3))
                         )
             
            n_cells[bundle_idx] = n_cell_i
            bundle_idx += 1
            logger.info(f"{int(n_cell_i)} cells of radius {round(1e6 * c_i_rad,3)} [um] will be placed in the voxel")
        return n_cells

    def _place_fiber_grid(self):
        R = Rotation.from_euler('Y', self.thetas, degrees = True)
        ymin   = -0.5 * self.buffer
        stride = (self.buffer + self.voxel_dimensions) / len(self.fiber_fractions)  
        if (self.n_fibers > 0).any():
            ctrs = [None] * self.n_fibers.shape[0]
            for i, n_fiber_i in enumerate(self.n_fibers):

                xs = np.linspace((-0.5*self.buffer)+max(self.fiber_radii), self.voxel_dimensions+(0.5*self.buffer)-max(self.fiber_radii), n_fiber_i)
                ys = xs[:]

                yv, xv = np.meshgrid(xs, ys)

                # Split the voxel into n_fiber_bundle parts along the y-axis
                yv_i = yv[np.logical_and( ymin <= yv, yv <= ymin + stride )]
                xv_i = xv[np.logical_and( ymin <= yv, yv <= ymin + stride )]
                
                # ith_bundle_fiber_centers
                ctrs_i = np.stack([xv_i, yv_i, np.zeros(xv_i.shape[0])], axis = -1)
               
                # Append to the array of total fiber centers
                ctrs[i] = ctrs_i  
                ymin += stride 

            # Rotate The Fiber Bundles
            ctrs_r = [np.einsum('ij, Fj -> Fi', 
                                 R.as_matrix()[n_fiber, :, :], 
                                 ctrs[n_fiber]
                                ) for n_fiber in range(self.n_fibers.shape[0])]

           
            # Calc Affine that aligns the middle point of each fiber bundle and
            # apply to the subsequent fiber bundle
            for i, ctrs_r_i_p1 in enumerate(ctrs_r[1:]):
                
                ctrs_r_i_p1_mid = np.median(ctrs_r_i_p1, axis= 0)
                ctrs_r_i_m1_mid = np.median(ctrs_r[i], axis = 0) 

                b = ctrs_r_i_p1_mid - ctrs_r_i_m1_mid
                b[1] = 0

                ctrs_r[i+1] -= b 
                
            # Instantiate the Fiber Objects    
            fibers = []
            for i, ctr_i in enumerate(ctrs_r):
                for f_i in range(ctr_i.shape[0]): 
                    fibers.append(
                        fiber(
                        center = ctr_i[f_i, :].astype(np.float32),
                        direction=R.as_matrix()[i, :, :].dot(np.array([0., 0., 1.])).astype(np.float32),
                        bundle=np.array(i),
                        step= np.sqrt(6.0 * self.fiber_diffusions[i] * self.dt).astype(np.float32),
                        radius=self.fiber_radii[i],
                        kappa=self.kappa[i],
                        L=np.array(self.voxel_dimensions, dtype=np.float32),
                        A=self.Amplitude[i],
                        P=self.Periodicity[i],
                        theta = np.array(self.thetas[i], dtype=np.float32)
                        )
                    )
            # get fiber boundary
            self.z_min = np.amin(np.concatenate(ctrs_r)[:, z])

        else:
            # If no fibers, instantiate a null fiber object with negative radius. 
            fibers = [fiber(
                        center  = np.zeros(3, dtype=np.float32),
                        direction   = np.zeros(3, dtype=np.float32),
                        bundle      = -1,
                        step        = np.array(0),
                        radius      = -1 * np.ones(1, dtype=np.float32),
                        kappa       = np.zeros(1, dtype=np.float32),
                        L           = np.array(self.voxel_dimensions, dtype=np.float32),
                        A           = np.zeros(1, dtype=np.float32),
                        P           = np.ones(1, dtype=np.float32),
                        theta = np.array(0., dtype=np.float32)
                        )]
        return fibers

    def _place_cell_lattice(self) -> List[Type[cell]] :
        if (self.n_cells > 0).any(): 
            k = 0   
            # Find the Boundary of the Voxel's resident microstructure
            
            if (self.n_fibers > 0).any():
                bdy_mins = np.concatenate([np.full(2, np.amin(self.fiber_radii)), np.full(1, np.amin(self.z_min))])
                bdy_maxs = np.concatenate([np.full(2, self.voxel_dimensions - bdy_mins[0]), np.full(1, bdy_mins[-1] + self.voxel_dimensions)])
            else:
                bdy_mins = np.zeros(3)
                bdy_maxs = np.full(3, self.voxel_dimensions)

            # initiate cell centers
            c_ctr = np.zeros(
                             (self.n_cells.sum(), 3),    
                             dtype= np.float32
                            )
            # collect cell radii
            r_i = np.concatenate([
                                  np.full(shape = (n_cell_i, ), fill_value=rad_i) 
                                  for n_cell_i, rad_i in zip(self.n_cells, self.cell_radii)
                                 ])
            c_k = 0
            MAX_ITER = 1000000
            while c_k < c_ctr.shape[0]:
                k += 1
                # Place the cell
                for dim, dim_min, dim_max in zip([x,y,z], bdy_mins, bdy_maxs):
                    c_ctr[c_k, dim] = np.random.uniform(low = dim_min + r_i[c_k], high = dim_max - r_i[c_k])

                # Calculate the distance between cell c_k and cell 0...c_k-1
                d_c_k_m_c_i_leq_k = np.linalg.norm(c_ctr[c_k, :] - c_ctr[0:c_k,:], ord = 2, axis = -1)

                # if cell c_k doesn't overlap with any of cells 0...c_k-1, place the cell by incrementing c_k
                if (d_c_k_m_c_i_leq_k > (r_i[c_k] + r_i[0:c_k])).all():
                    c_k += 1
                    k = 0 # reset the number of iterations per cell
                    sys.stdout.write('\r' + f"simDRIFT:simulate: Placed Cell [{c_k}/{c_ctr.shape[0]}]")
                    sys.stdout.flush()
        
                if k == MAX_ITER:
                    # Increment c_k one more. The first 0,...,c_k elements are non-zero
                    # so need to iterate over range(0, c_k + 1)
                    c_k += 1
                    sys.stdout.write('\n')
                    logger.info(
                        f"Tried to place cell {c_k} for {k} iterations! [{c_ctr.shape[0] - c_k}/{c_ctr.shape[0]}] Cells could not be placed.\n" \
                        f"The effective density is: {self._calc_effective_cell_density(c_ctr):.5f}%"
                        )
                    break
        
            cells = [None]*c_k
            # Instantiate Cell Objects
            current = 0
            for i, c_n_i in enumerate(self.n_cells):
                start, stop = current, current + c_n_i
                
                for j in range(start, min(stop, c_k)):        
                    cells[j] = cell(
                        cell_center=c_ctr[j, :],
                        cell_radius=self.cell_radii[i],
                        cell_bundle=i
                    )
                current = stop
            sys.stdout.write('\n')
        else:
            cells = [cell(
                cell_center=np.zeros(3, dtype=np.float32),
                cell_radius= -1*np.ones(1, dtype=np.float32),
                cell_bundle=-1
            )]
        
        return cells
    
    def _calc_effective_cell_density(self, c_ctr) -> float:
        v_cells = 0.
        v_voxel = self.voxel_dimensions**3
        
        current = 0
        for c_i_rad, c_i_n in zip(self.cell_radii, self.n_cells):
            start, stop = current, current + c_i_n
            n_c_i_eff = ( ~((c_ctr[start:stop, :] == 0).all(axis = -1))).sum()
            v_cells += n_c_i_eff*(4/3)*np.pi*c_i_rad**3 
            current = stop
        return v_cells / v_voxel

    def _place_spins(self):
        # Determine the voxel bdy. If there exist fibers, because of the rotations 
        # the voxel is bound by [0, voxel_dimensions]^{2} x [f_z_min, f_z_min + voxel_dimensions].
        # Without rotation the voxel is bound by [0, voxel_dimensions]^{3}.
        if (self.n_fibers > 0).any():
            bdy_mins = np.concatenate([np.full(2, np.amin(self.fiber_radii)), np.full(1, np.amin(self.z_min))])
            bdy_maxs = np.concatenate([np.full(2, self.voxel_dimensions - bdy_mins[0]), np.full(1, bdy_mins[-1] + self.voxel_dimensions)])
        else:
            bdy_mins = np.zeros(3)
            bdy_maxs = np.full(3, self.voxel_dimensions)

        # Place The Spins
        r0 = np.zeros((self.n_walkers, 3))
        for d, d_min, d_max in zip([x,y,z], bdy_mins, bdy_maxs):
            r0[:, d] = np.random.uniform(low = d_min - self.buffer, high = d_max - self.buffer, size = (self.n_walkers))

        # Intantiate The Spin Object
        spins = [spin( r0[ii, :].astype(np.float32)) for ii in range(r0.shape[0])] 
        return spins

def instantiate_geometry_obj(args):
    """Helper function to initiate relevant placement routines.
    """
    return VoxelGeometry(
        n_walkers = args.n_walkers,
        fiber_fractions=args.fiber_fractions,
        fiber_radii=args.fiber_radii,
        fiber_thetas=args.thetas,
        fiber_diffusions=args.fiber_diffusions,
        kappa=args.kappa,
        amplitude=args.Amplitude,
        periodicity=args.Periodicity,
        cell_fractions=args.cell_fractions,
        cell_radii=args.cell_radii,
        voxel_dimensions=args.voxel_dimensions,
        dt=args.dt,
        buffer=args.buffer,
        void_distance=args.void_distance
    )


