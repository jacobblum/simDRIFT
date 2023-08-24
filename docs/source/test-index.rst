*********************
Test Suite
*********************
The ``simDRIFT`` test suite will run 20 DWI forward simulations to verify that simDRIFT's core functionalities, physics, and output types and shapes are as expected. On an RTX3090, the test suite takes a few minutes to complete; however, this time will depend on your hardware.
As a remark, the DWI forward simulations run here are relatively small-scale, and thus you may not notice a significant increase in GPU utilization while the test suite runs.

Test Signals and Trajectories, 
=================================
* (1/20) Test Signal Types: assert that the forward simulated signal is a Nifti file.

* (2/20) Test Trajectory File Types: assert that the forward trajectory matrix is a .npy file.

* Test Signal Shapes: assert that the forward simulated signal shapes correspond to the input diffusion schemes. 
    
  * (4/20)-DBSI-99-Direction 
  * (5/20)-ABCD-103-Direction 
  * (6/20)-NODDI-145-Direction

* (7/20) Test Custom bval/bvec files: assert that the forward simulated signal induced by a custom, input-supplied bvec and bval file matches the shapes from the specified custom diffusion scheme.

* Test Trajectory Shapes: assert that the forward simulated trajectory matrix matches the size of the input number of spins in the ensemble. \

  * (8/20)-100 spins
  * (9/20)-256,000 spins
  * (10/20)-1,000,000 spins

Test Physics
====================================

* Test Water Physics: verify that the forward simulated water-only signal corresponds to a diffusion tensor matching the input water diffusivity parameter, and verify that this diffusion tensor is isotropic. 

  * (11/20)- D_water = 3.0 :math:`\mu m^{2} / ms` <-> AD = RD = 3.0  :math:`\mu m^{2} / ms`
  * (12/20)- D_water = 2.0 :math:`\mu m^{2} / ms` <-> AD = RD = 2.0  :math:`\mu m^{2} / ms`
  * (13/20)- D_water = 1.0 :math:`\mu m^{2} / ms` <-> AD = RD = 1.0  :math:`\mu m^{2} / ms`

* Test Fiber Physics: verify that the forward simulated respective fiber-only signals corresponds to a diffusion tensor matching the input fiber diffusivity parameter, and verify that this diffusion tensor is anisotropic. 
  
  * (14/20)- [D_fiber_1, D_fiber_2, D_fiber_3] = [1.0, 2.0, 2.0] ( :math:`\mu m^{2} / ms`), and AD :math:`>>` RD 
  * (15/20)- [D_fiber_1, D_fiber_2, D_fiber_3] = [1.0, 1.5, 2.0] ( :math:`\mu m^{2} / ms`), and AD :math:`>>` RD
  * (16/20)- [D_fiber_1, D_fiber_2, D_fiber_3] = [1.0, 1.0, 1.5] ( :math:`\mu m^{2} / ms`), and AD :math:`>>` RD

* Test Cell Physics: verify that the forward simulated cell-only signal, at various cell radii, corresponds to an isotropic diffusion tensor 

  * (17/20)- r = [1.0 :math:`\mu m`, 1.0 :math:`\mu m`], AD = RD
  * (18/20)- r = [1.5 :math:`\mu m`, 1.5 :math:`\mu m`], AD = RD   
  * (19/20)- r = [2.0 :math:`\mu m`, 2.0 :math:`\mu m`], AD = RD

* (20/20) Test Single Cell Physics: verify that the forward simulated cell-only signal, at various cell radii, corresponds to an isotropic diffusion tensor, r = [1.0 :math:`\mu m`], AD = RD
