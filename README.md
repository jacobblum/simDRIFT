
![This is an image](https://github.com/jacobblum/dMRI-MCSIM/blob/main/images/github_logo.PNG)
# dMRI SIM: Monte Carlo Simulation of diffusion MRI

## Diffusion Simulation
To sample the diffusion propogator in realistic manner on complex biological domains, we developed a Python 3.9 library to simulate the molecular self diffusion processes of spins and PGSE signal aquisition thereof. The spin diffusion process is modelled by an ensemble of random walkers, 
uniformly distrubted within the computational domain (an image voxel),   that are allowed to step within their local environment (Fibers, Cells, or the Extracellular/Extrafiber Matrix). Because the echo times of typical diffusion experiments are shorter than the intercellular preexchange lifetime of water, we are able to neglect any flux between the local environments. In particular, spins intially placed into fibers (cells, extracelluar/extrafiber matrix) cannot walk outside of fiber (cell, extracellular/extrafiber matrix), which is ensured by rejection sampling of steps in a random direction,  $\mathbf{u} \in S^{2}$, from the xoroshiro128+ psudorandom number generator:

$$
    \mathbf{r}_{i} = \mathbf{r}_{i-1} + \sqrt{6 \cdot \mathbf{D}_{0}^{\text{local}} \mathrm{d} t }  \cdot \mathbf{u} 
$$

Given the large number of spins required for convergence of the simulated PGSE signal, our code was developed with considerable 
attention towards preformance. In particular, individual spin trajectories are computed on a signle thread of the 
graphical processing unit (GPU), allowing for a non-linear dependence between the number of spins in the ensemble and the simulation runtime. 
Typical experiements feature between $256 \cdot 10^{3}$ to $1 \cdot 10^{6}$ radnom walkers, and are able to be completed within an hour depending on the 
complexity of the microstructure within the simulated imaging voxel. 

The diffusion process may be simpilated in arbitrary geometries admited by two fiber populations with set-able orientations, densities, and 
local diffusivities $\mathbf{D}_{0}^{\text{local}}$ and two cell populations with set-able radii and densitites.

## Signal Acquisition
Data from the simulated spin trajectories is then used to compute the echo signal admited by the following PGSE experiment
![This is an image](https://github.com/jacobblum/dMRI-MCSIM/blob/main/images/PGSE_sequence.png)

Standard numerical integration gives that the k-th dMRI signal, correspodning to the k-th diffusion gradient $g_{k}$ is given by:

$$E(\mathbf{g}_{k}, t = TE ) = \frac{1}{N_{\text{Walkers}}} \displaystyle \sum_{i = 1}^{N_{\text{Walkers}}} \exp \bigg ( -i \cdot \sum_{t}^{N_{t}} \gamma \mathbf{g}_{k}(t)^{T} \mathbf{r}(t) \mathrm{d} t \bigg)$$

This signal can then be used for validation of diffusion models. In particular, we are interested in evalutating a given signal processing model's ability 
to solve the inverse problem of recovering the ground truth intrinsic diffusivities of the local structures defined in our simulation of the forward problem: PGSE signal aquisition on complex biological domains. 
