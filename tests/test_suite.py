import sys 
import os 
import numpy as np
import pytest
import nibabel as nb 
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
import glob as glob

## Test Signals
@pytest.mark.parametrize("expected", [('.nii',)])
def test_signal_types(expected):
    """
    1. Check that the forward simulated signal is a Nifti file
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cmd = r"simDRIFT"
    cmd += f" simulate --n_walkers 256000 --fiber_fractions 0,0 --cell_fractions 0,0 --Delta .001 --voxel_dims 10 --buffer 0 --verbose no" 
    os.system(cmd)
    signals = glob.glob(os.getcwd() + os.sep + 'signals' + os.sep + '*')
    for signal in signals:
        assert os.path.splitext(signal)[-1:] == expected


@pytest.mark.parametrize("expected", [('.npy'),])
def test_trajectory_types(expected):
    """
    1. Check that the forward simulated trajectory is a .npy file
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cmd = r"simDRIFT"
    cmd += f" simulate --n_walkers 256000 --fiber_fractions 0,0 --cell_fractions 0,0 --Delta .001 --water_diffusivity {input} --voxel_dims 10 --buffer 0 --verbose no" 
    os.system(cmd)
    trajectories = glob.glob(os.getcwd() + os.sep + 'trajectories' + os.sep + '*')
    for trajectory in trajectories:
        assert os.path.splitext(trajectory)[-1:] == (expected,)

@pytest.mark.parametrize("input, expected", [('DBSI_99', 99), ('ABCD_102', 102), ('NODDI_145', 145)])
def test_signal_shapes(input, expected):
    """
    1. Check that the forward simulated signal matches the number of bvals and bvecs used in the 'imaging' experiment 
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cmd = r"simDRIFT"
    cmd += f" simulate --n_walkers 256000 --fiber_fractions 0,0 --cell_fractions 0,0 --Delta 1 --diff_scheme {input} --voxel_dims 10 --buffer 0 --verbose no" 
    os.system(cmd)
    signal = nb.load(os.getcwd() + os.sep + 'signals' + os.sep + 'water_signal.nii').get_fdata()  
    assert signal.shape == (expected,)

@pytest.mark.parametrize("input, expected", [((r'C:\Users\Jacob\Desktop\dMRI-MCSIM-Jacob-s-Version-Updated\Version_2.1.0\src\data\bval99', r'C:\Users\Jacob\Desktop\dMRI-MCSIM-Jacob-s-Version-Updated\Version_2.1.0\src\data\bvec99'), 99)])
def test_custom_diffusion_scheme(input, expected):
    """
    1. Check that the forward simulated signal matches the number of bvals and bvecs used in the PGSE experiment w/ 'custom' 
       bvals and bvecs (i.e., loaded in from a path)
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cmd = r"simDRIFT"
    cmd += f" simulate --n_walkers 256000 --fiber_fractions 0,0 --cell_fractions .1,.1 --Delta 0.001 --voxel_dims 20 --buffer 0 --verbose no"
    cmd += f" --bvals {input[0]} --bvecs {input[1]}"
    os.system(cmd)
    signal = nb.load(os.getcwd() + os.sep + 'signals' + os.sep + 'water_signal.nii').get_fdata()  
    assert signal.shape == (expected,)

@pytest.mark.parametrize("input, expected", [(99, 99), (256 * 1e3, 256 * 1e3), (1e6,1e6,)])
def test_trajectory_shapes(input, expected):
    """
    1. Check that the forward simulated trajectories match the number of simulated spins
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cmd = r"simDRIFT"
    cmd += f" simulate --n_walkers {int(input)} --fiber_fractions 0,0 --cell_fractions 0,0 --Delta 0.001 --voxel_dims 10 --buffer 0 --verbose no" 
    os.system(cmd)
    trajectories = np.load(os.getcwd() + os.sep + 'trajectories' + os.sep + 'total_trajectories_t2p.npy')
    assert trajectories.shape[0] == int((expected))

## Test Physics

sys.path.append(r"C:\Users\Jacob\Desktop\dMRI-MCSIM-Jacob-s-Version-Updated\Version_2.1.0")
from src.data import diffusion_schemes
bvals, bvecs = diffusion_schemes.get_from_default('DBSI_99')

# RMK: All Diffusivity units are in um^2 / ms 

@pytest.mark.parametrize("input, expected", [(3.0, 3.0), (1.0, 1.0), (2.0, 2.0)])
def test_water_physics(input, expected):
    """
    1. Check that the forward simulated water-only signal corresponds to a diffusion tensor matching the input water diffusivity
    2. Check that the forward simulated water-only signal corresponds to an isotropic diffusion tensor 
    """
    gtab = gradient_table(bvals, bvecs)
    tenmodel = dti.TensorModel(gtab)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cmd = r"simDRIFT"
    cmd += f" simulate --n_walkers 256000 --fiber_fractions 0,0 --cell_fractions 0,0 --Delta 1 --water_diffusivity {input} --voxel_dims 10 --buffer 0 --verbose no" 
    os.system(cmd)
    signal = nb.load(os.getcwd() + os.sep + 'signals' + os.sep + 'water_signal.nii').get_fdata()  
    tenfit = tenmodel.fit(signal)
    assert np.isclose(1e3 * tenfit.ad, expected, atol = .1) 
    assert np.isclose(1e3 * tenfit.ad, 1e3 * tenfit.rd, atol=.1)

@pytest.mark.parametrize("input, expected", [((1.0,2.0,2.5), (1.0,2.0,2.5)), ((1.0,1.5,2.0), (1.0,1.5,2.0)), ((1.0,1.0,1.5), (1.0,1.0,1.5))])
def test_fiber_physics_multi(input, expected):
    """
    1. Check that the forward simulated fiber-only signal corresponds to a diffusion tensor matching the input fiber diffusivities
    2. Check that the forward simulated fiber-only signal corresponds to an anisotropic diffusion tensor 
    """
    gtab = gradient_table(bvals, bvecs)
    tenmodel = dti.TensorModel(gtab)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cmd = r"simDRIFT"
    cmd += f" simulate --n_walkers 256000 --fiber_fractions .30,.30,.30 --fiber_diffusions {input[0]},{input[1]},{input[2]} --thetas 0,0,0 --fiber_radii 1,1,1 --cell_fractions 0,0 --Delta 1 --voxel_dims 30 --buffer 0 --verbose no" 
    os.system(cmd)

    fiber_1_signal = nb.load(os.getcwd() + os.sep + 'signals' + os.sep + 'fiber_1_signal.nii').get_fdata()
    fiber_2_signal = nb.load(os.getcwd() + os.sep + 'signals' + os.sep + 'fiber_2_signal.nii').get_fdata()
    fiber_3_signal = nb.load(os.getcwd() + os.sep + 'signals' + os.sep + 'fiber_3_signal.nii').get_fdata()   
    tenfit_1 = tenmodel.fit(fiber_1_signal)
    tenfit_2 = tenmodel.fit(fiber_2_signal)
    tenfit_3 = tenmodel.fit(fiber_3_signal)

    assert (np.isclose(1e3 * np.array([tenfit_1.ad, tenfit_2.ad, tenfit_3.ad]), np.array(expected), atol = 0.1).all())
    assert (np.array([tenfit_1.fa, tenfit_2.fa, tenfit_3.fa]) > .20).all()

@pytest.mark.parametrize("input, expected", [(1.0, 1.0)])
def test_fiber_physics_single(input, expected):
    gtab = gradient_table(bvals, bvecs)
    tenmodel = dti.TensorModel(gtab)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cmd = r"simDRIFT"
    cmd += f" simulate --n_walkers 256000 --fiber_fractions .30 --fiber_diffusions {input} --cell_fractions 0,0 --Delta 1 --voxel_dims 10 --buffer 0 --verbose no" 
    os.system(cmd)
    fiber_1_signal = nb.load(os.getcwd() + os.sep + 'signals' + os.sep + 'fiber_1_signal.nii').get_fdata()
    tenfit_1 = tenmodel.fit(fiber_1_signal)
    assert np.isclose(1e3 * tenfit_1.ad, np.array(expected), atol = 0.1)

@pytest.mark.parametrize("input", [(1.0,1.0), (1.5,1.5), (2.0,2.0)])
def test_cell_physics_multi(input):
    """
    1. Check that the forward simulated cell-only signal corresponds to an isotropic diffusion tensor
       RMK: the inverse problem measured diffusivity here will strongly depend on the diffusion time, thus, this test only requires that 
           the cell diffusion be isotropic  
    """
    gtab = gradient_table(bvals, bvecs)
    tenmodel = dti.TensorModel(gtab)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cmd = r"simDRIFT"
    cmd += f" simulate --n_walkers 256000 --fiber_fractions 0,0 --cell_fractions .1,.1 --cell_radii {input[0]},{input[1]} --Delta 1 --voxel_dims 20 --buffer 0 --verbose no" 
    os.system(cmd)
    signal = signal = nb.load(os.getcwd() + os.sep + 'signals' + os.sep + 'cell_signal.nii').get_fdata()  
    tenfit = tenmodel.fit(signal)
    assert np.isclose(1e3 * tenfit.ad, 1e3 * tenfit.rd, atol = 0.1) 

@pytest.mark.parametrize("input", [(1.0)])
def test_cell_physics_single(input):
    """
    1. Check that the forward simulated cell-only signal corresponds to an isotropic diffusion tensor
       RMK: the inverse problem measured diffusivity here will strongly depend on the diffusion time, thus, this test only requires that 
           the cell diffusion be isotropic  
    """
    gtab = gradient_table(bvals, bvecs)
    tenmodel = dti.TensorModel(gtab)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cmd = r"simDRIFT"
    cmd += f" simulate --n_walkers 256000 --fiber_fractions 0,0 --cell_fractions .1 --cell_radii {input} --Delta 1 --voxel_dims 20 --buffer 0 --verbose no" 
    os.system(cmd)
    signal = signal = nb.load(os.getcwd() + os.sep + 'signals' + os.sep + 'cell_signal.nii').get_fdata()  
    tenfit = tenmodel.fit(signal)
    assert np.isclose(1e3 * tenfit.ad, 1e3 * tenfit.rd, atol = 0.1) 
    

def run_tests():
    cmd = f"pytest {os.path.abspath(__file__)}"
    os.system(cmd)

  