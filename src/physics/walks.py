import torch
import logging
from src.physics.common import (
    _axis_oriented_distance, 
    _gamma_t,
    _d_gt__d_t
)

logger = logging.getLogger('simDRIFT')

def fiber_step(
                ith_spin_in_fiber   : torch.FloatTensor,
                r_t                 : torch.FloatTensor,
                jth_fiber_step      : torch.FloatTensor,
                jth_fiber_radii     : torch.FloatTensor,
                jth_fiber_center    : torch.FloatTensor,
                jth_fiber_direction : torch.FloatTensor,
                jth_fiber_kappa     : torch.FloatTensor,
                jth_fiber_L         : torch.FloatTensor,
                jth_fiber_A         : torch.FloatTensor,
                jth_fiber_P         : torch.FloatTensor,
                jth_fiber_theta     : torch.FloatTensor,
                MAX_ITER            : int = 500
                ) -> None:


    r_t_f     = r_t[ith_spin_in_fiber, :] 

    r_t_f_p_1   = torch.zeros(
                              r_t_f.shape, 
                              device = r_t.device,
                              dtype  = r_t.dtype
                              )
    
    converged = torch.zeros(
                            r_t_f.shape[0], 
                            dtype  = torch.bool, 
                            device = r_t.device
                            )

    k = 0
    while not converged.all():
        
        dir                    = torch.randn( size = ( (~converged).sum(), 3), device = r_t.device, dtype = r_t.dtype)             
        dir                   /= torch.linalg.norm(dir, dim = -1, ord = 2)[:, None]

        r_t_f_p_1[~converged, :] = r_t_f[~converged, :] + jth_fiber_step[~converged][:, None] * dir
    
        kth_iter_ith_spins_jth_fiber_dynamic_ctr = jth_fiber_center[~converged, :] + _gamma_t(
                                                                                                r_t_f_p_1[~converged], 
                                                                                                jth_fiber_direction[~converged], 
                                                                                                jth_fiber_theta[~converged], 
                                                                                                jth_fiber_kappa[~converged],
                                                                                                jth_fiber_L[~converged],
                                                                                                jth_fiber_A[~converged],
                                                                                                jth_fiber_P[~converged]
                                                                                                )
    
        kth_iter_ith_spins_jth_fiber_dynamic_normal = _d_gt__d_t(
                                                                    r_t_f_p_1[~converged], 
                                                                    jth_fiber_direction[~converged], 
                                                                    jth_fiber_theta[~converged], 
                                                                    jth_fiber_kappa[~converged],
                                                                    jth_fiber_L[~converged],
                                                                    jth_fiber_A[~converged],
                                                                    jth_fiber_P[~converged]
                                                                    )

        converged[~converged] = _axis_oriented_distance( 
                                                        x = r_t_f_p_1[~converged, :], 
                                                        y = kth_iter_ith_spins_jth_fiber_dynamic_ctr, 
                                                        v = kth_iter_ith_spins_jth_fiber_dynamic_normal
                                                        ) < jth_fiber_radii[~converged]

        k += 1

        if k > MAX_ITER:
            warning_str =  '\n' +\
                            'Spin(s): {} '.format(str((torch.argwhere(ith_spin_in_fiber > 0)[:, 0])[~converged])) + \
                            'failed to converge with MAX_ITER = {}. Please consider increasing MAX_ITER.'.format(MAX_ITER)
            
            if (~converged).sum() > 0: 
                logger.info(warning_str)
                break


    r_t[ith_spin_in_fiber, :] = r_t_f_p_1
    return           

def cell_step(ith_spin_in_cell    : torch.BoolTensor,
              r_t                 : torch.FloatTensor,
              jth_fiber_radii     : torch.FloatTensor,
              jth_fiber_center    : torch.FloatTensor,
              jth_fiber_direction : torch.FloatTensor,
              jth_fiber_kappa     : torch.FloatTensor,
              jth_fiber_L         : torch.FloatTensor,
              jth_fiber_A         : torch.FloatTensor,
              jth_fiber_P         : torch.FloatTensor,
              jth_fiber_theta     : torch.FloatTensor,
              jth_cell_center     : torch.FloatTensor,
              jth_cell_radii      : torch.FloatTensor,
              water_step          : torch.FloatTensor,
              MAX_ITER            : int = 500) -> None:
    
    r_t_c = r_t[ith_spin_in_cell]

    r_t_c_p_1 = torch.zeros(
                            r_t_c.shape,
                            device = r_t.device 
                            )
    
    converged = torch.zeros(
                            r_t_c.shape[0],
                            dtype  = torch.bool,
                            device = r_t.device
                            )
    k = 0
    while not converged.all():
    
        dir = torch.randn( size = ( (~converged).sum(), 3), device= r_t.device )             
        dir /= torch.linalg.norm(dir, dim = -1, ord = 2)[:, None]

        r_t_c_p_1[~converged, :] = r_t_c[~converged, :] + water_step*dir

        #####################################################################################
        #                           Check Distance to Fibers                                #
        #####################################################################################

        kth_iter_ith_spins_jth_fiber_dynamic_ctr = jth_fiber_center + _gamma_t(r_t_c_p_1[~converged], 
                                                                               jth_fiber_direction, 
                                                                                jth_fiber_theta, 
                                                                                jth_fiber_kappa,
                                                                                jth_fiber_L,
                                                                                jth_fiber_A,
                                                                                jth_fiber_P, 
                                                                                force_broadcast=True
                                                                                )
        
        kth_iter_ith_spins_jth_fiber_dynamic_normal = _d_gt__d_t(r_t_c_p_1[~converged], 
                                                                 jth_fiber_direction, 
                                                                 jth_fiber_theta, 
                                                                 jth_fiber_kappa,
                                                                 jth_fiber_L,
                                                                 jth_fiber_A,
                                                                 jth_fiber_P, 
                                                                 force_broadcast=True
                                                                )
        
        d_kth_iter_ith_spin_to_jth_fiber         = r_t_c_p_1[~converged, None, :] - kth_iter_ith_spins_jth_fiber_dynamic_ctr
        d_l_2_sqr_kth_iter_ith_spin_to_jth_fiber = torch.square(torch.linalg.norm(d_kth_iter_ith_spin_to_jth_fiber, axis = -1, ord = 2))

        p     = torch.einsum('NFi, NFi -> NF', 
                                    d_kth_iter_ith_spin_to_jth_fiber, 
                                    kth_iter_ith_spins_jth_fiber_dynamic_normal
                                    )

        p_sqr = torch.square(p)
        proj_dist_kth_iter_ith_spin_to_jth_fiber_along_p = torch.sqrt(d_l_2_sqr_kth_iter_ith_spin_to_jth_fiber - p_sqr)

        #####################################################################################
        #                           Check Distance to Cells                                 #
        #####################################################################################

        d_kth_iter_jth_spin_to_jth_cell         = r_t_c_p_1[~converged, :] - jth_cell_center[~converged, :]
        d_l_2_sqr_kth_iter_jth_spin_to_jth_cell = torch.linalg.norm(d_kth_iter_jth_spin_to_jth_cell, axis = -1, ord = 2)

        #####################################################################################
        #                             Check for Convergence                                 #
        #####################################################################################

        converged[~converged] = torch.stack(
                                            [
                                                (proj_dist_kth_iter_ith_spin_to_jth_fiber_along_p > jth_fiber_radii).all(dim = 1),
                                                (d_l_2_sqr_kth_iter_jth_spin_to_jth_cell          < jth_cell_radii[~converged])
                                            ], axis = -1
                                            ).all(axis = -1) # True if not in any fibers or cells. False  o.w.

        k += 1

        if k >= MAX_ITER:
            warning_str =  '\n' +\
                            'Spin(s): {} '.format(str((torch.argwhere(ith_spin_in_cell > 0)[:, 0])[~converged])) + \
                            'failed to converge with MAX_ITER = {}. Please consider increasing MAX_ITER.'.format(MAX_ITER)
            
            if (~converged).sum() > 0: 
                logger.info(warning_str)
                break

    r_t[ith_spin_in_cell, :] = r_t_c_p_1
    return




def water_step(ith_spin_in_water             : torch.BoolTensor,
                        r_t                  : torch.FloatTensor,
                        jth_fiber_radii      : torch.FloatTensor,
                        jth_fiber_center     : torch.FloatTensor,
                        jth_fiber_direction  : torch.FloatTensor,
                        jth_fiber_kappa      : torch.FloatTensor,
                        jth_fiber_L          : torch.FloatTensor,
                        jth_fiber_A          : torch.FloatTensor,
                        jth_fiber_P          : torch.FloatTensor,
                        jth_fiber_theta      : torch.FloatTensor,
                        jth_cell_centers     : torch.FloatTensor,
                        jth_cell_radii       : torch.FloatTensor,
                        water_step           : torch.FloatTensor,
                        MAX_ITER             : int = 500) -> None:
    
    r_t_w = r_t[ith_spin_in_water]

    r_t_w_p_1 = torch.zeros(
                            r_t_w.shape,
                            device = r_t.device,
                            dtype = r_t.dtype
                            )
    
    converged = torch.zeros(
                            r_t_w.shape[0],
                            dtype  = torch.bool,
                            device = r_t.device
                            )
    
    k = 0
    while not converged.all():
    
        dir = torch.randn( size = ( (~converged).sum(), 3), device= r_t.device, dtype=r_t.dtype )             
        dir /= torch.linalg.norm(dir, dim = -1, ord = 2)[:, None]

        r_t_w_p_1[~converged, :] = r_t_w[~converged, :] + water_step*dir

        #####################################################################################
        #                           Check Distance to Fibers                                #
        #####################################################################################

        kth_iter_ith_spins_jth_fiber_dynamic_ctr = jth_fiber_center + _gamma_t(r_t_w_p_1[~converged], 
                                                                                jth_fiber_direction, 
                                                                                jth_fiber_theta, 
                                                                                jth_fiber_kappa,
                                                                                jth_fiber_L,
                                                                                jth_fiber_A,
                                                                                jth_fiber_P, 
                                                                                force_broadcast=True
                                                                                )
        
        kth_iter_ith_spins_jth_fiber_dynamic_normal = _d_gt__d_t(r_t_w_p_1[~converged], 
                                                                    jth_fiber_direction, 
                                                                    jth_fiber_theta, 
                                                                    jth_fiber_kappa,
                                                                    jth_fiber_L,
                                                                    jth_fiber_A,
                                                                    jth_fiber_P, 
                                                                    force_broadcast=True
                                                                    )
        
        d_kth_iter_ith_spin_to_jth_fiber         = r_t_w_p_1[~converged, None, :] - kth_iter_ith_spins_jth_fiber_dynamic_ctr
        d_l_2_sqr_kth_iter_ith_spin_to_jth_fiber = torch.square(torch.linalg.norm(d_kth_iter_ith_spin_to_jth_fiber, axis = -1, ord = 2))

        p     = torch.einsum('NFi, NFi -> NF', 
                                    d_kth_iter_ith_spin_to_jth_fiber, 
                                    kth_iter_ith_spins_jth_fiber_dynamic_normal
                                    )

        p_sqr = torch.square(p)
        proj_dist_kth_iter_ith_spin_to_jth_fiber_along_p = torch.sqrt(d_l_2_sqr_kth_iter_ith_spin_to_jth_fiber - p_sqr)

        #####################################################################################
        #                           Check Distance to Cells                                 #
        #####################################################################################

        d_kth_iter_ith_spin_to_jth_cell         = r_t_w_p_1[~converged, None, :] - jth_cell_centers[None, :, :]
        d_l_2_sqr_kth_iter_ith_spin_to_jth_cell = torch.linalg.norm(d_kth_iter_ith_spin_to_jth_cell, axis = -1, ord = 2)
        
    
        #####################################################################################
        #                             Check for Convergence                                 #
        #####################################################################################

        converged[~converged] = torch.stack(
                                            [
                                                (proj_dist_kth_iter_ith_spin_to_jth_fiber_along_p > jth_fiber_radii).all(dim = 1),
                                                (d_l_2_sqr_kth_iter_ith_spin_to_jth_cell > jth_cell_radii).all(dim = 1)
                                            ], axis = -1
                                            ).all(axis = -1) # True if not in any fibers or cells. False  o.w.
        
        k += 1

        if k >= MAX_ITER:
            warning_str =  '\n' +\
                            'Spin(s): {} '.format(str((torch.argwhere(ith_spin_in_water > 0)[:, 0])[~converged])) + \
                            'failed to converge with MAX_ITER = {}. Please consider increasing MAX_ITER.'.format(MAX_ITER)
            
            if (~converged).sum() > 0: 
                logger.info(warning_str)
                break

    r_t[ith_spin_in_water, :] = r_t_w_p_1
    return 
