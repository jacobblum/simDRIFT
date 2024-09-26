import numpy as np  
import os  
import nibabel as nb
import logging
import torch 
from typing import Dict, Union

logger = logging.getLogger('simDRIFT')

   
def calc_signal(phase : torch.FloatTensor) -> torch.FloatTensor:
    signal = 1/phase.shape[0] * torch.nansum(torch.cos(phase), dim = 0)
    return signal
    
def calc_fiber_signals(args) -> Dict[str, torch.FloatTensor]:
    fiber_signals_dict = {}

    # setup iterables
    N_fb = [n_fb for n_fb in range(1, args.fibers_bundle.max() + 1)] 
    s_N_fb_names = [f'fiber_{n_fb}_signal' for n_fb in N_fb]

    for n_fb, s_n_fb_name in zip(N_fb, s_N_fb_names):
        # find which fibers are in the n_fb-th bundle
        f_nb = torch.argwhere(args.fibers_bundle == n_fb)
        # collect intra-fiber spins in the n_fb-th bunle
        spins_in_f_nb = torch.isin(
            args.ith_spin_in_jth_fiber_key,
            f_nb
        )        
        # calculate the signal induced by these spins
        fiber_signals_dict[s_n_fb_name] = calc_signal(args.ensemble_phase[spins_in_f_nb])

    # if there are any fibers, calculate the total fiber signal
    if N_fb:
        fiber_signals_dict['total_fiber_signal'] = calc_signal(args.ensemble_phase[args.ith_spin_in_jth_fiber_key > -1])

    return fiber_signals_dict

def calc_cell_signals(args) -> Dict[str, torch.FloatTensor]:
    cell_signals_dict = {} 

    # setup iterables
    N_cb = [n_cb for n_cb in range(1, args.cells_bundle.max() + 1)] 
    s_N_cb_names = [f'cell_{n_cb}_signal' for n_cb in N_cb]

    for n_cb, s_n_cb_name in zip(N_cb, s_N_cb_names):
        # find which cells are in the n_cb-th bundle
        c_nb = torch.argwhere(args.cells_bundle == n_cb)
        # collect intra-cell spins in the n_cb-th bunle
        spins_in_c_nb = torch.isin(
            args.ith_spin_in_jth_cell_key,
            c_nb
        )        
        # calculate the signal induced by these spins
        cell_signals_dict[s_n_cb_name] = calc_signal(args.ensemble_phase[spins_in_c_nb])

    # if there are any cells, calculate the total cell signal
    if N_cb:
        cell_signals_dict['total_cell_signal'] = calc_signal(args.ensemble_phase[args.ith_spin_in_jth_cell_key > -1])
   
    return cell_signals_dict

def calc_water_signals(args) -> Dict[str, torch.FloatTensor]:
    return {'water_signal' : calc_signal(args.ensemble_phase[args.water_key > -1])}

def calc_total_signal(args) -> Dict[str, torch.FloatTensor]:
    return {'total_signal' : calc_signal(args.ensemble_phase)}

def calc_and_save_signals(args):

    # setup directory to save signals to
    SIGNALS_DIRECTORY = os.path.join(
        args.sim_out_dir, 'signals'
    )

    if not os.path.exists(SIGNALS_DIRECTORY): 
        os.mkdir(SIGNALS_DIRECTORY)    

    logger.info(f"saving signals to : {SIGNALS_DIRECTORY}")

    # calculate the fiber signals
    fiber_signals = calc_fiber_signals(args)

    # calculate the cell signals
    cell_signals = calc_cell_signals(args)

    # calcualte the water signals
    water_signals = calc_water_signals(args)

    # calculate total signals
    total_signal = calc_total_signal(args)

    # save data as NIfTI image. 
    for k,v in {**fiber_signals, **cell_signals, **water_signals, **total_signal}.items():
        img = nb.Nifti1Image(
            dataobj=v.to('cpu').numpy(),
            affine = None    
        )
        nb.save(img, filename=os.path.join(SIGNALS_DIRECTORY, k + '.nii.gz'))
    return

