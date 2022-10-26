# dMRI-MCSIM

__gpuSimulation.py:__

gpuSimulation.py is a high preformance diffusion MRI Monte Carlo Simulation libarary with allows users to simulate molecular diffusion in a $[0 \mu m \text{ , } 50\mu m]^{3}$ imaging voxel with: 2 fiber bundles with set-able volume fractions, intrinsic diffusivities, and crossing angles, and non-overlapping cells of 2 radii, each with a volume fraction, which either aggregate in a 'void' between the two fiber bundles. 

For best preformance, we reccomend using 512 Threads per block, and simulating some interger multiple of 512 spins. My RTX 3090, as well as the V100 GPUs on our lab server, can launch, at most, 128 conccurent thread blocks, and so for the fastest performance, 512*128 spins is reccomended. The computation time will obviously depend on the scanning parameters, however, with 10 ms diffusion time, our a 512*128 spin simulation will take a few minutes. 200k spin simulations take around 10-15 minutes, and Million spin simulations should be expected to take one to two hours depending on fiber and cell density. 


![alt text](https://github.com/jacobblum/dMRI-MCSIM/blob/main/figures_for_mcsim/Void%20Configuration-3.png)

__Setting the intra-voxel tissue configuration:__

The tissue configuration is set using the config.ini setter code. 








