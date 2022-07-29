from re import I
import numpy as np 
import matplotlib.pyplot as plt
import scipy.io
import nnlsq_model
import nibabel as nb
import pandas as pd
import true_signal
from tqdm import tqdm 
import nnlsq_model 


def cleanSpins(trajectory_1, trajectory_2, spin_positions, spin_key):

 
    traj_1_in_voxel = []
    traj_2_in_voxel = []
    idxs = []

    trajectory_1 = trajectory_1[(spin_positions != 3) & (spin_positions != 0), :, :]
    trajectory_2 = trajectory_2[(spin_positions !=3 ) & (spin_positions != 0), :, :]

    for j in range(trajectory_1.shape[0]):
        t1j_min_pos, t1j_max_pos = np.amin(trajectory_1[j,:,:]), np.amax(trajectory_1[j,:,:])
        t2j_min_pos, t2j_max_pos = np.amin(trajectory_2[j,:,:]), np.amax(trajectory_2[j,:,:])

        if (t1j_min_pos >= 0 and t1j_max_pos <= 200) and (t2j_min_pos >= 0 and t2j_max_pos <= 200):
            traj_1_in_voxel.append(trajectory_1[j,:,:])
            traj_2_in_voxel.append(trajectory_2[j,:,:])
            idxs.append(j)

    outputarg1 = np.array(traj_1_in_voxel) 
    outputarg2 = np.array(traj_2_in_voxel)

    print(outputarg1.shape)

    # 2802
    # 2224

    # 3913 cells 
    # 12122 water

    return outputarg1, outputarg2

def signal(trajectory_1, trajectory_2, bmax, num_bvals, spin_positions, xyz):
    gamma = 42.58
    delta = 1
    Delta = 20
    target_bmax = bmax # s/mm^2 
    dt = 0.0010 #ms

    spin_loc_key = 4
    if xyz:
        num_bvals = num_bvals
        b_vals = np.geomspace(100,target_bmax, num_bvals)
        Gx =  np.sqrt(10**-3 *b_vals/( gamma**2 * delta**2*(Delta-delta/3)))
        unit_gradients = np.zeros((3*len(Gx),3))    
        for i in range(3):
            unit_gradients[i*num_bvals: (i+1)*num_bvals,i] = Gx 
    else:
        unit_gradients = np.loadtxt(r"/Users/jacobblum/Desktop/MCSIM-Jacob/CrossingFibers/ABCD/sub-NDARDD890AYU_ses-01_dwi_sub-NDARDD890AYU_ses-01_dwi (1).bvec").T
        b_vals = np.loadtxt(r"/Users/jacobblum/Desktop/MCSIM-Jacob/CrossingFibers/ABCD/sub-NDARDD890AYU_ses-01_dwi_sub-NDARDD890AYU_ses-01_dwi (3).bval")
    all_signal = np.zeros(unit_gradients.shape[0])
    traj1_vox, traj2_vox = cleanSpins(trajectory_1, trajectory_2, spin_positions, spin_loc_key)
    
    for i in tqdm(range(len(unit_gradients[:,0]))):
        if xyz:
             gradient_strength = unit_gradients[i,:]
        else:
            gradient_strength = np.sqrt((b_vals[i] * 10**-3)/(gamma**2*delta**2*(Delta-delta/3)))*unit_gradients[i,:]
        signal = 0
        for j in range(len(traj1_vox[:,0])):
            spin_i_traj_1 = traj1_vox[j,:,:]
            spin_i_traj_2 = traj2_vox[j,:,:]
            phase_shift = gamma * np.sum(gradient_strength.dot((spin_i_traj_2-spin_i_traj_1).T)) * dt     
            signal = signal + np.exp( -1 * (0+1j) * phase_shift)
       
        signal = signal/traj1_vox.shape[0]
        all_signal[i] = np.abs(signal)


    dwi_img = np.zeros((1,1,1,all_signal.shape[0]))
    dwi_img[0,0,0,:] = all_signal

    nifti_dwi = nb.Nifti1Image(dwi_img, affine = np.eye(4))
    nb.save(nifti_dwi, r"/Volumes/LaCie/MCSIM-Jacob/CrossingFibers/Data/from117/22_0714/pure_fiber_signal" + ".nii")
    return all_signal, b_vals, traj1_vox.shape[0]

def plot():
    
    xyz = False 
    
  

    traj11 = np.load(r"/Volumes/LaCie/MCSIM-Jacob/CrossingFibers/Data/from117/22_0714/taj1_3_fibers_50k_no_crossing_fixed_old.npy")
    traj21 = np.load(r"/Volumes/LaCie/MCSIM-Jacob/CrossingFibers/Data/from117/22_0714/traj2_3_fibers_50k_no_crossing_fixed_old.npy")
    positions1 = np.load(r"/Volumes/LaCie/MCSIM-Jacob/CrossingFibers/Data/from117/22_0714/spin_positions_3_fibers_50k_no_crossing_fixed_old.npy")

    traj12 = np.load(r"//Volumes/LaCie/MCSIM-Jacob/CrossingFibers/Data/from116/22_0714/taj1_3_fibers_50k_no_crossing_fixed_old.npy")
    traj22 = np.load(r"/Volumes/LaCie/MCSIM-Jacob/CrossingFibers/Data/from116/22_0714/traj2_3_fibers_50k_no_crossing_fixed_old.npy")
    positions2 = np.load(r"/Volumes/LaCie/MCSIM-Jacob/CrossingFibers/Data/from116/22_0714/spin_positions_3_fibers_50k_no_crossing_fixed_old.npy")


    positions = np.hstack((positions1, positions2))

    
    traj1 = np.vstack((traj11, traj12))
    traj2 = np.vstack((traj21, traj22))

    print(traj1.shape )
    print(traj2.shape)

    
    all_signal, b_vals, num_spins = signal(traj1, traj2, 1500, 20, positions, xyz)

    if xyz:
        sx, sy, sz = all_signal[0:20], all_signal[20:40], all_signal[40:60]
    
        Dx, e_x = nnlsq_model.lstq(b_vals, sx)
        Dy, e_y = nnlsq_model.lstq(b_vals, sy)
        Dz, e_z = nnlsq_model.lstq(b_vals, sz)

        fig, (ax1, ax2) = plt.subplots(2,1, figsize = (10,12))
        ax1.plot(b_vals, np.log(sx), 'rx', label = r'$ADC^{x}:$ %s, %s' %(str(round(Dx[1],3)), str(round(e_x,5))))
        ax1.plot(b_vals, np.log(sy), 'bx', label = r'$ADC^{y}:$ %s, %s' %(str(round(Dy[1],3)), str(round(e_x,5))))
        ax1.plot(b_vals, np.log(sz), 'gx', label = r'$ADC^{z}$ %s, %s' %(str(round(Dz[1],3)),  str(round(e_x,5))))
        ax1.set_xlabel(r'$b_{k} \quad s/mm^{2}$')
        ax1.set_ylabel(r'$s_{k}$')
        ax1.set_title(r' dMRI Signal Attenuation - Total Signal')
        #ax1.set_ylim(bottom = 0, top =1.1)


        cs_x, d_x, e_x = nnlsq_model.nnlsq_model(b_vals, sx)
        cs_y, d_y, e_y = nnlsq_model.nnlsq_model(b_vals, sy)
        cs_z, d_z, e_z = nnlsq_model.nnlsq_model(b_vals, sz)
    
    
    
    
        ax2.vlines(d_x, ymin = np.zeros(d_x.shape[0]), ymax = cs_x, color = 'r', label = r'$s_{k}^{x}$: %s' %str(round(e_x,5)))
        ax2.vlines(d_y, ymin = np.zeros(d_y.shape[0]), ymax = cs_y, color = 'b', label = r'$s_{k}^{y}$: %s' %str(round(e_y,5)))
        ax2.vlines(d_z, ymin = np.zeros(d_z.shape[0]), ymax = cs_z, color = 'g', label = r'$s_{k}^{z}$: %s' %str(round(e_z,5)))
        ax2.set_xlabel(r' D $ \mu m^{2} / ms $')
        ax2.set_ylabel(r'Diffusivity Fraction')
        ax2.set_title(r'Diffusion Spectra')
        ax2.set_ylim(bottom = 0,top = 1.1)
        ax2.set_xlim(left = -.05, right = 3.6)
        ax1.legend()
        ax2.legend(loc = 'center')
        plt.show()
    return 

plot()