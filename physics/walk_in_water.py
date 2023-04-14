import numpy as np
import numba
from numba import jit, cuda
from jp import random, linalg
import math
import operator


@numba.cuda.jit(nopython=True, parallel=True)
def _diffusion_in_water(i,
                        random_states,
                        fiber_centers,
                        fiber_directions,
                        fiber_radii,
                        cell_centers,
                        cell_radii,
                        spin_positions,
                        step):

    previous_position = cuda.local.array(shape=3, dtype=numba.float32)
    proposed_new_position = cuda.local.array(shape=3, dtype=numba.float32)

    u3 = cuda.local.array(shape=3, dtype=numba.float32)

    for j in range(u3.shape[0]):
        u3[j] = 1/math.sqrt(3) * 1.

    invalid_step = True
    while invalid_step:

        stepped_into_fiber = False
        stepped_into_cell = False
        proposed_new_position = random.random_on_S2(random_states,
                                                    proposed_new_position,
                                                    i
                                                    )

        for k in range(proposed_new_position.shape[0]):
            previous_position[k] = spin_positions[i, k]
            proposed_new_position[k] = previous_position[k] + (step*proposed_new_position[k])

        for k in range(fiber_centers.shape[0]):
            dFv = linalg.dL2(proposed_new_position, fiber_centers[k, :], fiber_directions[k, :], True)
            if dFv < fiber_radii[k]:
                stepped_into_fiber = True
                break

        if not (stepped_into_fiber):
            for k in range(cell_centers.shape[0]):
                dC = linalg.dL2(proposed_new_position, cell_centers[k, :], u3, False)
                if dC < cell_radii[k]:
                    stepped_into_cell = True
                    break

        if operator.and_(not (stepped_into_cell), not (stepped_into_fiber)):
            invalid_step = False

    for k in range(proposed_new_position.shape[0]):
        spin_positions[i, k] = proposed_new_position[k]
    return
