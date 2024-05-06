import numpy as np
import sys
import random
import logging
import src.setup.spin_init_positions as spin_init_positions
import src.setup.objects as objects
from src.jp import linalg
from src.setup.mesh import VoxelSurfaceMesh

def _set_num_fibers(fiber_fractions, fiber_radii, voxel_dimensions, buffer, fiber_configuration):
    """Calculates the requisite number of fibers for the supplied fiber densities (volume fractions).

    :param fiber_fractions: User-supplied fiber densities (volume fractions)
    :type fiber_fractions: float, tuple
    :param fiber_radii: User-supplied fiber radii, in units of :math:`{\mathrm{μm}}`.
    :type fiber_radii: float, tuple
    :param voxel_dimensions: User-supplied voxel side length, in units of :math:`{\mathrm{μm}}`.
    :type voxel_dimensions: float
    :param buffer: User-supplied additional length to be added to the voxel size for placement routines, in units of :math:`{\mathrm{μm}}`.
    :type buffer: float
    :param fiber_configuration: Desired fiber geometry class name.
    :type fiber_configuration: str
    :return: List of grid sizes, float 
    :rtype: int, tuple
    """ 
    logging.info('------------------------------')
    logging.info(' Fiber Setup')
    logging.info('------------------------------') 

    num_fibers = []
    for i in range(len(fiber_fractions)):

        vl = (voxel_dimensions + buffer) ** 2
        num_fiber = int(np.sqrt( len(fiber_fractions) * ( vl * fiber_fractions[i])/(np.pi*fiber_radii[i]**2)))  
        num_fibers.append(num_fiber)
        
        logging.info(' {} fibers of type {} (R{} = {} (um))'.format(int(num_fibers[i]**2),int(i),int(i), 1e6 * fiber_radii[i]))
    logging.info(' Fiber geometry: {}'.format(fiber_configuration))

    return num_fibers

def _set_num_cells(cell_fraction, cell_radii, voxel_dimensions, buffer):
    """Calculates the requisite number of cells for the supplied cell densities (volume fractions).

    :param cell_fraction: User-supplied cell densities (volume fractions).
    :type cell_fraction: float, tuple
    :param cell_radii: User-supplied cell radii, in units of :math:`{\mathrm{μm}}`.
    :type cell_radii: float, tuple
    :param voxel_dimensions: User-supplied voxel side length, in units of :math:`{\mathrm{μm}}`.
    :type voxel_dimensions: float
    :param buffer: User-supplied additional length to be added to the voxel size for placement routines.
    :type buffer: float
    :return: List containing the number of each cell type.
    :rtype: float, tuple
    """

    logging.info('------------------------------')
    logging.info(' Cells Setup')
    logging.info('------------------------------')    
    num_cells = []
    for i in range(len(cell_fraction)):
        if cell_fraction[i] > 0:
            num_cells.append(int(
                (0.5*cell_fraction[i]*(voxel_dimensions**3)/((4.0/3.0)*np.pi*cell_radii[i]**3))))
        else:
            num_cells.append(int(0))
        logging.info(' {} cells with radius = {} um'.format(num_cells[i], cell_radii[i]))
    return num_cells

def _place_fiber_grid(self):

    

    """Routine for populating fiber grid within the simulated imaging voxel
    
    :param fiber_fractions: User-supplied fiber densities (volume fractions)
    :type fiber_fractions: float, tuple
    :param fiber_radii: Radii of each fiber type
    :type fiber_radii: float, tuple
    :param fiber_diffusions: User-supplied diffusivities for each fiber type
    :type fiber_diffusions: float, tuple
    :param thetas: Desired alignment angle for each fiber type, relative to :math:`{\\vu{z}}`
    :type thetas: float, tuple
    :param voxel_dimensions: User-supplied voxel side length, in units of :math:`{\mathrm{μm}}`
    :type voxel_dimensions: float
    :param buffer: User-supplied additional length to be added to the voxel size for placement routines, in units of :math:`{\mathrm{μm}}`
    :type buffer: float
    :param void_distance: Length of region for excluding fiber population, in units of :math:`{\mathrm{μm}}`
    :type void_distance: float
    :param fiber_configuration: Desired fiber geometry class. See `Class Objects`_ for further information.
    :type fiber_configuration: str
    :return: Class object containing fiber attributes. See `Class Objects`_ for further information.
    :rtype: object
    """
    num_fibers = _set_num_fibers(self.fiber_fractions, 
                                 self.fiber_radii, 
                                 self.voxel_dimensions, 
                                 self.buffer,
                                 self.fiber_configuration)

    rotation_matrices = linalg.Ry(self.thetas)
    fibers = []
    ymin   = -0.5 * self.buffer
    stride = (self.buffer + self.voxel_dimensions) / len(self.fiber_fractions)  

    total_ctrs = []
    for i in range(len(self.fiber_fractions)):
        ith_bundle_ctrs = []
        yv, xv = np.meshgrid(np.linspace((-0.5*self.buffer)+max(self.fiber_radii), self.voxel_dimensions+(0.5*self.buffer)-max(self.fiber_radii), num_fibers[i]),
                             np.linspace((-0.5*self.buffer)+max(self.fiber_radii), self.voxel_dimensions+(0.5*self.buffer)-max(self.fiber_radii), num_fibers[i]))
        
        for ii in range(yv.shape[0]):
            for jj in range(yv.shape[1]):
                fiber_cfg_bools = {'Penetrating': True,
                                   'Interwoven' : True,
                                   'Void'       : np.logical_or(xv[ii, jj] <= np.median(yv[0,:]) - 0.5 * self.void_distance, xv[ii, jj] > np.median(yv[0,:]) + 0.5 * self.void_distance)} 
                if np.logical_and( ymin <= yv[ii,jj], yv[ii,jj] <= ymin + stride ):         
                    if fiber_cfg_bools[self.fiber_configuration]:        
                        ith_bundle_ctrs.append(np.array([xv[ii,jj], yv[ii, jj], 0]))       
      
        total_ctrs.append(np.array(ith_bundle_ctrs))       
        ymin += stride 

    if self.fiber_configuration == 'Interwoven':
    # Select Fibers for Rotation if the Inter-Woven configuration is selected
        fiber_ctrs_regrouped = [[] for ii in range(len(self.fiber_fractions))]
        fiber_centers_linear = np.stack([center for Nfiber in range(len(self.fiber_fractions)) for center in total_ctrs[Nfiber]], axis = 0)
        for Y_index, Y in enumerate(np.unique(fiber_centers_linear[:, 1])):
            for fiber_index in [idx for idx in np.where(fiber_centers_linear[:, 1] == Y)[0]]:
                fiber_ctrs_regrouped[Y_index % len(self.fiber_fractions)].append(fiber_centers_linear[fiber_index])
        total_ctrs[:] = fiber_ctrs_regrouped[:]

    total_ctrs_prime = [np.einsum('ij, Fj -> Fi', rotation_matrices[Nfiber, :, :], total_ctrs[Nfiber]) for Nfiber in range(len(self.fiber_fractions))]

    mXp = []
    mZp = []
    
    # Align the Fibers
    for Nfiber in range(len(total_ctrs_prime)):
        
        mXp.append(np.median(total_ctrs_prime[Nfiber][:,  0]))
        mZp.append(np.median(total_ctrs_prime[Nfiber][:, -1]))

        if Nfiber > 0:
            Delta_mXp = mXp[Nfiber - 1] - mXp[Nfiber]
            Delta_mZp = mZp[Nfiber - 1] - mZp[Nfiber] 

            total_ctrs_prime[Nfiber][:,  0] += Delta_mXp
            total_ctrs_prime[Nfiber][:, -1] += Delta_mZp
            
        mXp[Nfiber] = np.median(total_ctrs_prime[Nfiber][:,  0])
        mZp[Nfiber] = np.median(total_ctrs_prime[Nfiber][:, -1])

    As = np.array([0.01, -0.01, -10.0]).astype(np.float32)*1e-6

    # Instantiate the Fiber Objects 
    for Nfiber in range(len(total_ctrs_prime)):
        for fiber in range(total_ctrs_prime[Nfiber].shape[0]):
            fibers.append(objects.fiber(center      = total_ctrs_prime[Nfiber][fiber, :],
                                        direction   = rotation_matrices[Nfiber, :, :].dot(np.array([0., 0., 1.])),
                                        bundle      = Nfiber,
                                        diffusivity = self.fiber_diffusions[Nfiber],
                                        radius      = self.fiber_radii[Nfiber],
                                        kappa       = 1.0,
                                        L           = self.voxel_dimensions,
                                        A           = As[Nfiber],
                                        P           = 1.0
                                        )
                        )
   
    # If no fibers, instantiate a null fiber object with negative radius. 
    if not fibers:

        fibers.append(objects.fiber(center      = np.zeros(3),
                                    direction   = np.zeros(3),
                                    bundle      = 0,
                                    diffusivity = 0,
                                    radius      = -1.0,
                                    kappa       = self.kappa,
                                    L           = self.voxel_dimensions + 15,
                                    A           = As[Nfiber],
                                    P           = self.P
                                    )
                    )
    return fibers

def _place_cells(self):
    """Routine for populating cells within the simulated imaging voxel
    
    :param fibers: Class object containing fiber attributes. See `Class Objects`_ for further information.
    :type fibers: object
    :param cell_radii: Radii of each cell type, in units of :math:`{\mathrm{μm}}`
    :type cell_radii: float, tuple
    :param cell_fractions: User-supplied densities (volume fractions) for each cell type
    :type cell_fractions: float, tuple
    :param fiber_configuration: Desired fiber geometry class. See `Class Objects`_ for further information.
    :type fiber_configuration: str
    :param voxel_dimensions: User-supplied voxel side length, in units of :math:`{\mathrm{μm}}`
    :type voxel_dimensions: float
    :param buffer: User-supplied additional length to be added to the voxel size for placement routines, in units of :math:`{\mathrm{μm}}`
    :type buffer: float
    :param void_distance: Length of region for excluding fiber population, in units of :math:`{\mathrm{μm}}`
    :type void_distance: float
    :param water_diffusivity: The user-supplied diffusivity for free water, in units of :math:`{\mathrm{μm}^2}\\, \mathrm{ms}^{-1}`.
    :type water_diffusivity: float
    :return: Class object containing cell attributes. See `Class Objects`_ for further information.
    :rtype: object
    """
    logging.info('------------------------------')
    logging.info(' Placing Cells...')
    logging.info('------------------------------')

    cell_centers_total = []
    num_cells = _set_num_cells(self.cell_fractions, self.cell_radii, self.voxel_dimensions, self.buffer)

    zmin = min([fiber.center[2] for fiber in self.fibers])
    zmax = zmin + self.voxel_dimensions

    if self.fiber_configuration == 'Void':
        ## Note[KLU]: Adjusted the regions below to be symmetric about the middle of the voxel 
        regions = np.array([[0-(self.buffer/2), self.voxel_dimensions+(self.buffer/2), 0.5*(self.voxel_dimensions - self.void_dist), 0.5*(self.voxel_dimensions + self.void_dist), zmin, zmax],
                            [0-(self.buffer/2), self.voxel_dimensions+(self.buffer/2), 0.5*(self.voxel_dimensions - self.void_dist), 0.5*(self.voxel_dimensions + self.void_dist), zmin, zmax]])
    else:
        regions = np.array([[0-(self.buffer/2), self.voxel_dimensions+(self.buffer/2), 0-(self.buffer/2), 0.5*self.voxel_dimensions, zmin, zmax],
                            [0-(self.buffer/2), self.voxel_dimensions+(self.buffer/2), 0.5*self.voxel_dimensions, self.voxel_dimensions+(self.buffer/2), zmin, zmax]])

    for i in (range(len(num_cells))):
        cellCenters = np.zeros((num_cells[i], 4))
        for j in range(cellCenters.shape[0]):
            if i == 0:
                sys.stdout.write('\r' + 'dMRI-SIM:  ' + str(j+1) + '/' + str(sum(num_cells)) + ' cells placed')
                sys.stdout.flush()
            else:
                sys.stdout.write('\r' + 'dMRI-SIM:  ' + str(num_cells[0]+(j+1)) + '/' + str(sum(num_cells)) + ' cells placed')
                sys.stdout.flush()
            if j == 0:
                invalid = True
                while (invalid):
                    radius = self.cell_radii[i]
                    xllim, xulim = regions[i, 0], regions[i, 1]
                    yllim, yulim = regions[i, 2], regions[i, 3]
                    zllim, zulim = regions[i, 4], regions[i, 5]
                    cell_x = np.random.uniform(xllim + radius, xulim - radius)
                    cell_y = np.random.uniform(yllim + radius, yulim - radius)
                    cell_z = np.random.uniform(zllim + radius, zulim - radius)
                    cell_0 = np.array([cell_x, cell_y, cell_z, radius])
                    proposedCell = cell_0
                    ctr = 0
                    if i == 0:
                        cellCenters[j, :] = proposedCell
                        invalid = False
                    elif i > 0:
                        for k in range(cell_centers_total[0].shape[0]):
                            distance = np.linalg.norm(
                                proposedCell-cell_centers_total[0][k, :], ord=2)
                            if distance < (radius + cell_centers_total[0][k, 3]):
                                ctr += 1
                                break
                    if ctr == 0:
                        cellCenters[j, :] = proposedCell
                        invalid = False
            elif (j > 0):
                invalid = True
                while (invalid):
                    xllim, xulim = regions[i, 0], regions[i, 1]
                    yllim, yulim = regions[i, 2], regions[i, 3]
                    zllim, zulim = regions[i, 4], regions[i, 5]
                    radius = self.cell_radii[i]
                    cell_x = np.random.uniform(xllim + radius, xulim - radius)
                    cell_y = np.random.uniform(yllim + radius, yulim - radius)
                    cell_z = np.random.uniform(zllim + radius, zulim - radius)
                    proposedCell = np.array([cell_x, cell_y, cell_z, radius])
                    ctr = 0
                    for k in range(j):
                        distance = np.linalg.norm(
                            proposedCell-cellCenters[k, :], ord=2)
                        if distance < 2*radius:
                            ctr += 1
                            break
                        if i > 0:
                            for l in range(cell_centers_total[0].shape[0]):
                                distance = np.linalg.norm(
                                    proposedCell-cell_centers_total[0][l, :], ord=2)
                                if distance < (radius + cell_centers_total[0][l, 3]):
                                    ctr += 1
                                    break
                    if ctr == 0:
                        cellCenters[j, :] = proposedCell
                        invalid = False
        cell_centers_total.append(cellCenters)
    output_arg = np.vstack([cell_centers_total[i] for i in range(len(cell_centers_total))])

    cells = []
    
    if not (output_arg).any():
        cells.append(objects.cell(cell_center=np.array([0., 0., 0.]), cell_radius=-1, cell_diffusivity=0.))
    else:
        for i in range(output_arg.shape[0]):
            cells.append(objects.cell(cell_center = output_arg[i,0:3], cell_radius=self.cell_radii[0], cell_diffusivity = self.water_diffusivity))
        sys.stdout.write('\n')
    return cells

def _place_spins(self):
    """Routine for randomly populating spins in the imaging voxel following a uniform probability distribution

    :param n_walkers: User-specified number of spins to simulate
    :type n_walkers: int
    :param voxel_dims: User-supplied voxel side length, in units of :math:`{\mathrm{μm}}`
    :type voxel_dims: float
    :param fibers: Class object ``objects.fibers`` containing fiber attributes. See `Class Objects`_ for further information.
    :type fibers: object
    :return: Class object ``objects.spins`` containing spin attributes. See `Class Objects`_ for further information.
    :rtype: object
    """
    fiber_ctrs = np.stack([fiber.center for fiber in self.fibers])
    bdyXmin = np.amin(fiber_ctrs[:,0])
    bdyXmax = np.amax(fiber_ctrs[:,0])
    bdyYmin = np.amin(fiber_ctrs[:,1])
    bdyYmax = np.amax(fiber_ctrs[:,1])
    bdyZmin = np.amin(fiber_ctrs[:,2])
    bdyZmax = np.amax(fiber_ctrs[:,2])

    spin_positions_t1m = np.vstack([np.random.uniform(low = bdyXmin,   high = bdyXmax, size = self.n_walkers),
                                    np.random.uniform(low = bdyYmin,   high = bdyYmax, size = self.n_walkers),
                                    np.random.uniform(low = bdyZmin,   high = bdyZmin + self.voxel_dimensions, size = self.n_walkers)]
                                    )

    spins = [objects.spin(spin_positions_t1m[:,ii]) for ii in range(spin_positions_t1m.shape[1])] 
    return spins

def setup(self):
    """Helper function to initiate relevant placement routines.
    """
    self.fibers = _place_fiber_grid(self)
    self.cells = _place_cells(self)
    self.spins = _place_spins(self)
    spin_init_positions._find_spin_locations(self)
    return

