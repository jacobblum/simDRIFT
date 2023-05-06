
![This is an image](https://github.com/jacobblum/dMRI-MCSIM/blob/main/images/github_logo.PNG)
# dMRI SIM: Monte Carlo Simulation of diffusion MRI

## Diffusion Simulation
To realistically sample the diffusion propagator on complex biological domains, we developed a Python 3.9+ library to simulate the molecular self diffusion processes of spins and PGSE signal acquisition. The spin diffusion process is modeled by an ensemble of random walkers, 
uniformly distributed within the computational domain (an image voxel),   that are allowed to step within their local environment (Fibers, Cells, or the Extracellular/Extrafiber Matrix). Because the echo times of typical diffusion experiments are shorter than the intercellular pre-exchange lifetime of water, we can neglect any flux between the local environments. In particular, spins initially placed into fibers (cells, extracellular/extra-axonal matrix) cannot walk outside of fiber (cell, extracellular/extra-axonal matrix), which is ensured by rejection sampling of steps in a random direction,  $\mathbf{u} \in S^{2}$, from the xoroshiro128+ pseudorandom number generator:

$\mathbf{r}_{i} = \mathbf{r}_{i-1} + \sqrt{6 \cdot \mathbf{D}_{0}^{\text{local}} \mathrm{d} t }  \cdot \mathbf{u}$

Given the large number of spins required for simulated PGSE signal convergence, our code was developed with considerable attention to performance. In particular, individual spin trajectories are computed on a single thread of the graphical processing unit (GPU), creating a non-linear dependence between the number of spins in the ensemble and the simulation runtime. 
Typical experiments feature between $256 \cdot 10^{3}$ to $1 \cdot 10^{6}$ random walkers and can be completed within an hour depending on the complexity of the simulated microstructure within the imaging voxel. 

The diffusion process may be simulated in arbitrary geometries with up to two distinct fiber populations (with user-adjustable orientations, radii, volume densities, and local diffusivities $\mathbf{D}_{0}^{\text{local}}$) and up to two distinct cell populations (with user-adjustable radii and volume densities)

## Signal Acquisition
Data from the simulated spin trajectories are then used to compute the echo signal as described by the following PGSE experiment:
![This is an image](https://github.com/jacobblum/dMRI-MCSIM/blob/main/images/PGSE_sequence.png)

Standard numerical integration gives that the $k\textsuperscript{th}$ dMRI signal, correspodning to the k-th diffusion gradient $g_{k}$ is given by:

$$E(\mathbf{g}_{k}, t = TE ) = \frac{1}{N_{\text{Walkers}}} \displaystyle \sum_{i = 1}^{N_{\text{Walkers}}} \exp \bigg ( -i \cdot \sum_{t}^{N_{t}} \gamma \mathbf{g}_{k}(t)^{T} \mathbf{r}(t) \mathrm{d} t \bigg)$$

This signal can then be used for the validation of diffusion models. In particular, we are interested in evaluating a given signal-processing model's ability to solve the inverse problem of recovering the ground truth intrinsic diffusivities of the local structures defined in our simulation of the forward problem: PGSE signal acquisition on complex biological domains.
