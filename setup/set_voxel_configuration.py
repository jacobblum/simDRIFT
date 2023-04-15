import numpy as np
import jp as jp
import sys
from jp import linalg
import objects
import matplotlib.pyplot as plt
import logging

def _set_num_fibers(fiber_fractions, fiber_radii, voxel_dimensions, buffer):
    num_fibers = []
    for i in range(len(fiber_fractions)):
        num_fiber = int(np.sqrt(
            (fiber_fractions[i] * (voxel_dimensions + buffer)**2)/(np.pi*fiber_radii[i]**2)))
        num_fibers.append(num_fiber)
    logging.info('------------------------------')
    logging.info('Num Fibers')
    logging.info('------------------------------') 
    logging.info('{} fibers of type 1'.format(str(num_fibers[0])))
    logging.info('{} fibers of type 2'.format(str(num_fibers[1])))
    return num_fibers


def _set_num_cells(cell_fraction, cell_radii, voxel_dimensions, buffer):
    num_cells = []
    for i in range(len(cell_radii)):
        if cell_fraction[i] > 0:
            num_cells.append(int(
                (0.5*cell_fraction[i]*((voxel_dimensions+buffer)**3)/((4.0/3.0)*np.pi*cell_radii[i]**3))))
        else:
            num_cells.append(int(0))
    logging.info('------------------------------')
    logging.info('Num Cells')
    logging.info('------------------------------')    
    logging.info('{} cells with radius = {} um'.format(str(num_cells[0]), str(cell_radii[0])))   
    logging.info('{} cells with radius = {} um'.format(str(num_cells[1]), str(cell_radii[1])))
    return num_cells


def _place_fiber_grid(fiber_fractions, fiber_radii, fiber_diffusions, thetas, voxel_dimensions, buffer, void_distance, fiber_configuration):

    num_fibers = _set_num_fibers(
        fiber_fractions, fiber_radii, voxel_dimensions, buffer)

    rotation_matrices = linalg.Ry(thetas)

    fibers = []
    for i in range(len(fiber_fractions)):

        yv, xv = np.meshgrid(np.linspace(0+max(fiber_radii), voxel_dimensions+buffer-max(fiber_radii), num_fibers[i]),
                             np.linspace(0+max(fiber_radii), voxel_dimensions+buffer-max(fiber_radii), num_fibers[i]))

        for ii in range(yv.shape[0]):
            for jj in range(yv.shape[1]):

                fiber_cfg_bools = {'Penetrating': True,
                                   'Void': np.logical_or(yv[ii, jj] > 0.5*(voxel_dimensions + buffer - void_distance), yv[ii, jj] < 0.5 * (voxel_dimensions+buffer + void_distance))
                                   }

                if np.logical_and(yv[ii, jj] >= (i)*0.5*(voxel_dimensions+buffer), yv[ii, jj] < (i+1)*0.5*(voxel_dimensions + buffer)):
                    if fiber_cfg_bools[fiber_configuration]:
                        fibers.append(objects.fiber(center=linalg.affine_transformation(xv, xv[ii, jj], yv[ii, jj], thetas, i),
                                                    direction=rotation_matrices[i, :, :].dot(
                                                        np.array([0., 0., 1.])),
                                                    bundle=i,
                                                    diffusivity=fiber_diffusions[i],
                                                    radius=fiber_radii[i]))

    return fibers


def _place_cells(num_cells, fibers, cell_radii, fiber_configuration, voxel_dimensions, buffer, void_dist):

    cell_centers_total = []

    zmin = min([fiber._get_center()[2] for fiber in fibers])
    zmax = max([fiber._get_center()[2] for fiber in fibers])

    if fiber_configuration == 'Void':
        regions = np.array([[0, voxel_dimensions+buffer, 0.5*(voxel_dimensions+buffer)-0.5*void_dist, 0.5*(voxel_dimensions+buffer)+0.5*void_dist, zmin, zmax],
                            [0, voxel_dimensions+buffer, 0.5*(voxel_dimensions+buffer)-0.5*void_dist, 0.5*(voxel_dimensions+buffer)+0.5*void_dist, zmin, zmax]])
    else:
        regions = np.array([[0, voxel_dimensions+buffer, 0, 0.5*(voxel_dimensions+buffer), zmin, zmax],
                            [0, voxel_dimensions+buffer, 0.5*(voxel_dimensions+buffer), voxel_dimensions+buffer, zmin, zmax]])

    logging.info('------------------------------')
    logging.info('Placing Cells...')
    logging.info('------------------------------')
    for i in (range(len(num_cells))):
        cellCenters = np.zeros((num_cells[i], 4))
        for j in range(cellCenters.shape[0]):
            if i == 0:
                sys.stdout.write('\r' + 'dmri-sim: ' + str(j+1) + '/' + str(num_cells[0]+num_cells[1]) + ' cells placed')
                sys.stdout.flush()
            else:
                sys.stdout.write('\r' + 'dmri-sim: ' + str(num_cells[0]+(j+1)) + '/' + str(num_cells[0]+num_cells[1]) + ' cells placed')
                sys.stdout.flush()
            if j == 0:
                invalid = True
                while (invalid):
                    radius = cell_radii[i]
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
                    radius = cell_radii[i]
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
    output_arg = np.vstack([cell_centers_total[0], cell_centers_total[1]])

    cells = []
    
    for i in range(output_arg.shape[0]):
        cells.append(objects.cell(cell_center = output_arg[i,0:3], cell_radius=2))
    sys.stdout.write('\n')
    return cells

def _place_spins(n_walkers: int, voxel_dims: float, fibers: object):
    spin_positions_t1m = np.vstack([np.random.uniform(low=0, high=voxel_dims, size=n_walkers),
                                    np.random.uniform(low=0, high=voxel_dims, size=n_walkers),
                                    np.random.uniform(low=min([fiber._get_center()[2] for fiber in fibers]),
                                                      high=max([fiber._get_center()[2] for fiber in fibers]),
                                                      size=n_walkers)])
    
    spins = [objects.spin(spin_positions_t1m[:,ii]) for ii in range(spin_positions_t1m.shape[1])]

    return spins




