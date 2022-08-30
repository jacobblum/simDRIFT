from cProfile import label
from tkinter import PROJECTING
import numpy as np 
import numba 
from numba import jit, cuda, int32, void, float32
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math
import jp
import matplotlib.pyplot as plt
import time 
import os  
from tqdm import tqdm
import nibabel as nb
import glob as glob 
import configparser
from ast import literal_eval
from multiprocessing import Process

class dmri_simulation:
    def __init__(self):
        numSpins = 0.0 
        numFibers = 0.0
        fiberFraction = 0.0
        fiberRadius = 0.0 #um
        Delta = 0.0 #ms
        dt = 0.0 #ms
        delta = 0.0 #ms
        numCells = 0.0
        cellFraction = 0.0
        cellRadii = 0.0 #um
        spinPositionsT1m = 0
        fiberPositionsT1m = 0
        fiberPositionsT2p = 0
        cellPositionsT1m = 0
        cellPositionsT2p = 0
        extraPositionsT1m = 0
        extraPositionsT2p = 0
        spinInFiber_i = 0
        spinInCell_i = 0
        fiberRotationReference = 0
        Thetas = 0
        fiberDiffusions  = 0
        rotMat = 0
        simulateFibers = 0
        simulateCells = 0
        simulateExtra = 0
        fiberConfiguration = ''
        buffer = 100
        bvals = 0
        bvecs = 0
        return
    
    def set_parameters(self, numSpins, fiberFraction, fiberRadius, Thetas, fiberDiffusions, cellFraction, cellRadii, fiberConfiguration, Delta, dt, voxelDim, buffer, path_to_bvals, path_to_bvecs):
        self.bvals = np.loadtxt(path_to_bvals) 
        self.bvecs = np.loadtxt(path_to_bvecs)
        self.voxelDims = voxelDim
        self.buffer = buffer 
        self.numSpins = numSpins
        self.fiberFraction = fiberFraction
        self.fiberRadius = fiberRadius
        self.Thetas = Thetas
        self.fiberDiffusions = fiberDiffusions
        self.fiberRotationReference = self.rotation()        
        self.numFibers = self.set_num_fibers()
        self.cellFraction = cellFraction
        self.cellRadii = cellRadii
        self.fiberCofiguration = fiberConfiguration
        self.numCells = self.set_num_cells()
        self.Delta = Delta
        self.dt = dt
        self.delta = dt
        self.fiberCenters = self.place_fiber_grid()
        self.cellCenters = self.place_cell_grid()
        self.spinPotionsT1m = np.random.uniform(low = 0 + self.buffer*.5, high = self.voxelDims+0.5*(self.buffer), size = (int(self.numSpins),3))
        self.spinInFiber_i = -1*np.ones(self.numSpins)
        self.spinInCell_i = -1*np.ones(self.numSpins)
        self.get_spin_locations()

    def from_config(self, path_to_configuration_file):

        
        ## Simulation Parameters
        config = configparser.ConfigParser()
        config.read(path_to_configuration_file)
        numSpins = literal_eval(config['Simulation Parameters']['numSpins'])
        fiberFraction = literal_eval(config['Simulation Parameters']['fiberFraction'])
        fiberRadius = literal_eval(config['Simulation Parameters']['fiberRadius'])
        Thetas = literal_eval(config['Simulation Parameters']['Thetas'])
        fiberDiffusions = literal_eval(config['Simulation Parameters']['fiberDiffusions'])
        cellFraction = literal_eval(config['Simulation Parameters']['cellFraction'])
        cellRadii = literal_eval(config['Simulation Parameters']['cellRadii'])
        fiberConfiguration = (config['Simulation Parameters']['fiberConfiguration'])
        simulateFibers = literal_eval(config['Simulation Parameters']['simulateFibers'])
        simulateCells = literal_eval(config['Simulation Parameters']['simulateCells'])
        simulateExtra = literal_eval(config['Simulation Parameters']['simulateExtraEnvironment'])

        ## Scanning Parameters
        Delta = literal_eval(config['Scanning Parameters']['Delta'])
        dt = literal_eval(config['Scanning Parameters']['dt'])
        voxelDims = literal_eval(config['Scanning Parameters']['voxelDim'])
        buffer = literal_eval(config['Scanning Parameters']['buffer'])
        bvals_path = config['Scanning Parameters']['path_to_bvals']
        bvecs_path = config['Scanning Parameters']['path_to_bvecs']

        ## Saving Parameters
        path_to_save = config['Saving Parameters']['path_to_save_file_dir']

        self.set_parameters(
            numSpins=numSpins,
            fiberFraction= fiberFraction,
            fiberRadius=fiberRadius,
            Thetas=Thetas,
            fiberDiffusions=fiberDiffusions,
            cellFraction=cellFraction,
            cellRadii=cellRadii,
            fiberConfiguration=fiberConfiguration,
            Delta=Delta,
            dt=dt,
            voxelDim=voxelDims,
            buffer=buffer,
            path_to_bvals= bvals_path, 
            path_to_bvecs= bvecs_path)
        self.simulate(
            simulateFibers=simulateFibers,
            simulateCells=simulateCells,
            simulateExtraEnvironment=simulateExtra)
                
        self.save_data(path_to_save, plot_xyz=True)
        return

    def set_num_fibers(self):
        fiberFraction = self.fiberFraction
        fiberRadius = self.fiberRadius
        numFibers = []
        for i in range(len(fiberFraction)):
            numFiber = int(np.sqrt((fiberFraction[i] * (self.voxelDims+self.buffer)**2)/(np.pi*fiberRadius**2)))
            numFibers.append(numFiber)    
        print('FIBER GRID: \n {}'.format(numFibers))
        return numFibers
    
    def set_num_cells(self):
        cellFraction = self.cellFraction 
        numCells = []
        for i in range(len(self.cellRadii)):
            cellRadius = self.cellRadii[i]
            numCells.append(int(np.cbrt((cellFraction*((self.voxelDims+self.buffer)**3)/((4/3)*np.pi*cellRadius**3))))) 
        return numCells
    
    def place_fiber_grid(self):
        outputCords = []
        for i in range(len(self.fiberFraction)):
            fiberCordinates = np.zeros(((self.numFibers[i])**2,6))
            fiberYs, fiberXs = np.meshgrid(np.linspace(0+self.fiberRadius,self.voxelDims+self.buffer-self.fiberRadius, self.numFibers[i]), 
                                           np.linspace(0+self.fiberRadius,self.voxelDims+self.buffer-self.fiberRadius, self.numFibers[i]))
            fiberCordinates[:,0] = fiberXs.flatten()
            fiberCordinates[:,1] = fiberYs.flatten()
            fiberCordinates[:,3] = self.fiberRadius
            idx = np.where((fiberCordinates[:,1] >= i*0.5*(self.voxelDims+self.buffer)) & (fiberCordinates[:,1] < (i+1)*0.5*(self.voxelDims+self.buffer)))[0]
            outputCords.append(fiberCordinates[idx, :])
        outputArg =  np.vstack([outputCords[0], outputCords[1]])

        if self.fiberCofiguration == 'Inter-Woven':
            Ys_mod2 = np.unique(outputArg[:,1])[::2]
            idx = (np.in1d(outputArg[:,1], Ys_mod2))
            fiberCordinates_pre_rotation = outputArg[idx, 0:3]

        if self.fiberCofiguration == 'Penetrating':
            idx = np.where(outputArg[:,1] < 0.5*(self.voxelDims+self.buffer))[0]
            fiberCordinates_pre_rotation = outputArg[idx, 0:3]
        
        if self.fiberCofiguration == 'Non-Penetrating':
            fiber_idx = np.where( (((outputArg[:,0] > 0) & (outputArg[:,0] < (0.5)*(self.voxelDims+self.buffer))) & ((outputArg[:,1] > 0) & (outputArg[:,1] < (0.5)*(self.voxelDims+self.buffer))))
                          | (((outputArg[:,0] > 0.5*(self.voxelDims+self.buffer)) & (outputArg[:,0] < (self.voxelDims+self.buffer))) & ((outputArg[:,1] > 0.5*(self.voxelDims+self.buffer)) & (outputArg[:,1] < (self.voxelDims+self.buffer))))
                            )[0]
            outputArg = outputArg[fiber_idx]  
            idx = np.where((((outputArg[:,0] > 0) & (outputArg[:,0] < (0.5)*(self.voxelDims+self.buffer))) & ((outputArg[:,1] > 0) & (outputArg[:,1] < (0.5)*(self.voxelDims+self.buffer)))))[0]
            fiberCordinates_pre_rotation = outputArg[idx, 0:3]
            

        rotatedCords = (self.rotMat.dot(fiberCordinates_pre_rotation.T)).T
        if rotatedCords.shape[0] > 0:
            z_correct = np.amin(rotatedCords[:,2]) # Want the grid to be placed at z = 0
            rotatedFibers = rotatedCords 
            rotatedFibers[:,2] = rotatedFibers[:,2] + np.abs(z_correct )
            outputArg[idx, 0:3], outputArg[idx, 4], outputArg[idx, 5], outputArg[[i for i in range(outputArg.shape[0]) if i not in idx],5] = rotatedFibers, 1, self.fiberDiffusions[0], self.fiberDiffusions[1] 
        
        return outputArg
    
    def place_cell_grid(self):
        print('PLACING CELLS')
        numCells = self.numCells
        cellCentersTotal = []
        if self.fiberCofiguration == 'Non-Penetrating':
            regions = np.array([[0,self.voxelDims+self.buffer,0,0.5*(self.voxelDims+self.buffer),0.5*(self.voxelDims+self.buffer), self.voxelDims+self.buffer], 
                                [0,0.5*(self.voxelDims+self.buffer),0.5*(self.voxelDims+self.buffer), self.voxelDims+self.buffer,0,self.voxelDims+self.buffer]])
        else: 
            regions = np.array([[0,self.voxelDims+self.buffer,0,0.5*(self.voxelDims+self.buffer), 0,self.voxelDims+self.buffer],
                                [0,self.voxelDims+self.buffer,0.5*(self.voxelDims+self.buffer),self.voxelDims+self.buffer,0,self.voxelDims+self.buffer]])
        
        for i in range(len(numCells)):
            print('{} / {}'.format(i+1, len(numCells)))
            cellCenters = np.zeros((numCells[i]**3, 4))
            for j in tqdm(range(cellCenters.shape[0])):
                xllim, xulim = regions[i,0], regions[i,1]
                yllim, yulim = regions[i,2], regions[i,3]
                zllim, zulim = regions[i,4], regions[i,5]
                radius = self.cellRadii[i]
                cell_x = np.random.uniform(xllim + radius, xulim - radius)
                cell_y = np.random.uniform(yllim + radius, yulim - radius)
                cell_z = np.random.uniform(zllim + radius, zulim - radius)
                cell_j = np.array([cell_x, cell_y, cell_z, radius])
                
                ## Ensures that the randomly placed cells are non-overlapping 
                ctr = 0
                if j > 0:
                    for k in range(j):
                        distance = np.linalg.norm(cell_j[0:3] - cellCenters[k,0:3]) 
                        if distance < (cell_j[3] + cellCenters[k,3]):
                            ctr += 1
                            break                       
                if ctr == 0:
                    cellCenters[j,:] = cell_j
            cellCentersTotal.append(cellCenters)
        return np.vstack([cellCentersTotal[0], cellCentersTotal[1]])
    
    def get_spin_locations(self):
        print('Getting location of {} spins'.format(self.numSpins))
        spinInFiber_i_GPU = self.spinInFiber_i.astype(np.float32)
        spinInCell_i_GPU  = self.spinInCell_i.astype(np.float32)
        spinInitialPositions_GPU = self.spinPotionsT1m.astype(np.float32)
        fiberCenters_GPU = self.fiberCenters.astype(np.float32)
        cellCenters_GPU = self.cellCenters.astype(np.float32)
        Start = time.time()
        self.find_spin_locations.forall(self.numSpins)(spinInFiber_i_GPU, spinInCell_i_GPU, spinInitialPositions_GPU, fiberCenters_GPU, cellCenters_GPU, self.fiberRotationReference)
        End = time.time()
        print('Finding {} spins - task completed in {} sec'.format(self.numSpins, End-Start))
        self.spinInFiber_i = spinInFiber_i_GPU
        self.spinInCell_i = spinInCell_i_GPU
    
    def rotation(self):        
        rotationReferences = np.zeros((len(self.Thetas),3))
        for i,theta in enumerate(self.Thetas):
            theta = np.radians(theta)
            s, c = np.sin(theta), np.cos(theta)
            Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            self.rotMat = Ry
            z = np.array([0,0,1])
            rotationReferences[i,:] = Ry.dot(z)
        print('Fiber Rotation Matrix: \n {}'.format(rotationReferences))
        return rotationReferences


    def simulate(self, simulateFibers, simulateCells, simulateExtraEnvironment):        
        self.simulateFibers = simulateFibers
        self.simulateCells = simulateCells
        self.simulateExtra = simulateExtraEnvironment
        

        """
        Simulate Fiber Diffusion: 
        """
        if self.simulateFibers:
            self.fiberPositionsT1m = self.spinPotionsT1m[self.spinInFiber_i != -1].astype(np.float32)
            fiberSpins = cuda.to_device(self.spinPotionsT1m[self.spinInFiber_i != -1].astype(np.float32))
            fiberAtSpin_i = self.spinInFiber_i[self.spinInFiber_i != -1].astype(np.float32)
            rng_states_fibers = create_xoroshiro128p_states(len(self.spinInFiber_i[self.spinInFiber_i != -1]), seed = 42)
            print('STARTING FIBER SIMULATION')
            Start = time.time()
            self.diffusion_in_fiber.forall(len(fiberAtSpin_i))(rng_states_fibers,
                                    fiberSpins,
                                    fiberAtSpin_i,
                                    int(self.Delta/self.dt),
                                    self.fiberCenters.astype(np.float32),
                                    self.fiberRotationReference,
                                    self.dt)
            End = time.time()
            self.fiberPositionsT2p = fiberSpins.copy_to_host()
            print('ENDING FIBER SIMULATION')
            print(fiberSpins.shape)
            print('Fiber Diffusion Compuation Time: {} seconds'.format(End-Start))
            
        """
        Simulate Intra-Cellular Diffusion: 
            Each Spin must know which cell it is in before distributing computation to the GPU. Also, to avoid looping over all of the fibers, we need to also pass an array of penetrating fiber indicies to diffusion_in_cells
        """
        
        if self.simulateCells:
            cellSpins = self.spinPotionsT1m[(self.spinInCell_i > -1) & (self.spinInFiber_i < 0)] 
            self.cellPositionsT1m = cellSpins.copy()
            cellSpins_GPU = (cellSpins)
            cellAtSpin_i_GPU = (self.spinInCell_i[(self.spinInCell_i > -1) & (self.spinInFiber_i < 0)])
            rng_states_cells = create_xoroshiro128p_states(cellSpins.shape[0], seed = 42)
            print('STARTING CELLULAR SIMULATION')
            Start = time.time()
            self.diffusion_in_cell.forall(cellSpins.shape[0])(rng_states_cells, 
                                    cellSpins_GPU,
                                    cellAtSpin_i_GPU,
                                    int(self.Delta/self.dt),
                                    self.cellCenters,
                                    self.fiberCenters.astype(np.float32),
                                    self.fiberRotationReference,
                                    self.dt
            )
            End = time.time()
            print('ENDING CELLULAR SIMULATION')
            self.cellPositionsT2p = cellSpins_GPU
            print('Cell Diffusion Computation Time: {} seconds'.format(End - Start))
        
        
        """
        Simulate Extra-Cellular and Extra-Axonal Diffusion
        """
        if self.simulateExtra:
            self.extraPositionT1m = (self.spinPotionsT1m[(self.spinInCell_i < 0) & (self.spinInFiber_i < 0)])
            rng_states_Extra = create_xoroshiro128p_states(self.extraPositionT1m.shape[0], seed = 42)
            extraSpins_GPU = cuda.to_device(self.extraPositionT1m.astype(np.float32))
            print('STARTING WATER SIMULATION')
            Start = time.time()
            self.diffusion_in_water.forall(self.extraPositionT1m.shape[0])(rng_states_Extra,
                                            extraSpins_GPU,
                                            int(self.Delta/self.dt),
                                            self.cellCenters,
                                            self.fiberCenters.astype(np.float32),
                                            self.fiberRotationReference,
                                            self.dt
            )
            End = time.time()
            print('ENDED WATER SIMULATION')
            self.extraPositionT2p = extraSpins_GPU.copy_to_host()
            print('Water Diffusion Computation Time: {} seconds'.format(End - Start))
            
    def find_penetrating_fibers(self, cellCenters, fiberCenters, fiberRotationReference):
        
        """
        Global Variables:

        penetratingFibers - (numCells, numPotentialPenetratingFibers); the [i,j] element of this array is the index of a penetrating fiber [READ/WRITE]
        cellCenters - (numCells, 4); x,y,z, radius of cell[i]
        fiberCenters - (numFibers, 4); x,y,z, radius of fiber[i]

        Local Variables
        indexList - (25, ) the index of penetrating fibers 
        ctr - indexList index variable
        distanceFiber - distance from fiber to cell center
        """
        penetratingFibersList = []
     
        for i in range(cellCenters.shape[0]):
            penetratingFibersSublist = []
            for j in range(fiberCenters.shape[0]):
                distance = math.sqrt(np.linalg.norm(cellCenters[i,0:3]-fiberCenters[j,0:3], ord = 2)**2 - ((cellCenters[i,0:3]-fiberCenters[j,0:3]).dot(fiberRotationReference))**2)
                if distance < cellCenters[i,3]:
                    penetratingFibersSublist.append(j)
            penetratingFibersList.append(penetratingFibersSublist)
        return penetratingFibersList       

    @cuda.jit 
    def find_spin_locations(spinInFiber_i, spinInCell_i, initialSpinPositions, fiberCenters, cellCenters, fiberRotationReference):
        i = cuda.grid(1)
        if i > initialSpinPositions.shape[0]:
            return 
        
        """ Find the position of each of the ensemble's spins within the imaging voxel

            Parameters
            ----------
            spinInFiber_i: 1-d ndarray
                The index of the fiber a spin is located within; -1 if False, 0... N_fibers if True.
            spinInCell_i: 1-d ndarray
                The index of the cell a spin is located within; -1 if False, 0...N_cells if True
            initialSpinPositions : N_spins x 3 ndarray
                The initialSpinPositions[i,:] are the 3 spatial positions of the spin at its initial position
            fiberCenters: N_{fibers} x 6 ndarray
                The spatial position, rotmat index, intrinsic diffusivity, and radius of the i-th fiber
            cellCenters: N_{cells} x 4 ndarray
                The 3-spatial dimensions and radius of the i-th cell
            fiberRotationReference: 2 x 3 ndarray
                The Ry(Theta_{i}).dot([0,0,1]) vector  

            Returns
            -------
            spinInFiber_i : 1-d ndarray
                See parameters note
            spinInCell_i: 1-d ndarray
                See parameters note             

            Notes
            -----
            None

            References
            ----------
            None

            Examples
            --------
            >>> self.find_spin_locations.forall(self.numSpins)(spinInFiber_i_GPU, spinInCell_i_GPU, spinInitialPositions_GPU, fiberCenters_GPU, cellCenters_GPU, self.fiberRotationReference)
        """
        KeyFiber = int32(-1)
        KeyCell = int32(-1)
        spinPosition = cuda.local.array(shape = 3, dtype = float32)
        fiberDistance = float32(0.0)
        cellDistance = float32(0.0)
        rotationIndex = 0

        for k in range(spinPosition.shape[0]): spinPosition[k] = initialSpinPositions[i,k]
        for j in range(fiberCenters.shape[0]): 
            rotationIndex = int(fiberCenters[j,4])
            fiberDistance = jp.euclidean_distance(spinPosition, fiberCenters[j,0:3], fiberRotationReference[rotationIndex,:], 'fiber')
            if fiberDistance < fiberCenters[j,3]: 
                KeyFiber = j
                break
        
        for j in range(cellCenters.shape[0]):
            cellDistance = jp.euclidean_distance(spinPosition, cellCenters[j,0:3], fiberRotationReference[0,:], 'cell')
            if cellDistance < cellCenters[j,3]:
                KeyCell = j
                break
        
        cuda.syncthreads()
        spinInCell_i[i] = KeyCell
        spinInFiber_i[i] = KeyFiber
        cuda.syncthreads()
        return
            
    @cuda.jit
    def diffusion_in_fiber(rng_states, spinTrajectories, fiberIndexAt_i, numSteps, fiberCenters, fiberRotationReference, dt):   
        i = cuda.grid(1)
        if i > spinTrajectories.shape[0]:
            return
        """"
        Simulate molecular diffusion of spins within the fibers; Dirichlet boundary conditions 

        Parameters
        ----------
        rng_states : 1-d ndarray
            The random states of the spins
        spinTrajectories :  N_{fiber_spins} x 3 ndarray
            The spinTrajectories[i,:] sub-array are the spins physical cordinates
        spinInFiber_i: 1-d ndarray
            The index of the fiber a spin is located within; -1 if False, 0... N_fibers if True.
        numSteps: int
            int(Delta/dt); the number of timesteps
        fiberCenters: N_{fibers} x 6 ndarray
            The spatial position, rotmat index, intrinsic diffusivity, and radius of the i-th fiber
        fiberRotationReference: 2 x 3 ndarray
            The Ry(Theta_{i}).dot([0,0,1]) vector  
        dt : float32
            The time-discretization parameter

        Returns
        -------
        spinTrajectories: N_{fiber_spins} x 3 ndarray  
            for 0 <= step <= numSteps, the spins trajectory is updated until the spin steps within the fiber.           

        Notes
        -----
        re-stepping (currently implemented) vs. stepping is an ongoing discussion between me and S.K. Song about what is more reasonable. Tentatively, re-stepping allows for 
        accurate simulation at lower temporal resolutions, which gives favorable preformance results. 
        
        References
        ----------
        Discussions with S.K. Song about what physics is reasonable to implement here. 

        """
        
        i = cuda.grid(1)
        if i > spinTrajectories.shape[0]:
            return
        inx = int32(fiberIndexAt_i[i])
        D = float32(fiberCenters[inx, 5])
        Step = float32(math.sqrt(6*D*dt))
        prevPosition = cuda.local.array(shape = 3, dtype= float32)
        newPosition = cuda.local.array(shape = 3, dtype = float32)
        distanceFiber = float32(0.0)
        rotationIndex = int(fiberCenters[inx, 4])

        for step in range(numSteps):
            distance = fiberCenters[inx, 3] + .10
            while(distance > fiberCenters[inx, 3]):
                newPosition = jp.randomDirection(rng_states, newPosition, i)
                for k in range(newPosition.shape[0]): 
                    prevPosition[k] = spinTrajectories[i,k] 
                    newPosition[k] = prevPosition[k] + (Step * newPosition[k])
                distance = jp.euclidean_distance(newPosition,fiberCenters[inx,0:3], fiberRotationReference[rotationIndex,:], 'fiber')
            #if distance > fiberCenters[inx,3]:
                #for k in range(newPosition.shape[0]): newPosition[k] = prevPosition[k]
            cuda.syncthreads()
            for k in range(newPosition.shape[0]): spinTrajectories[i,k] = newPosition[k]
            cuda.syncthreads()
        return

    @cuda.jit
    def diffusion_in_cell(rng_states, spinTrajecotires, cellAtSpin_i, numSteps, cellCenters, fiberCenters, fiberRotationReference, dt):
        i = cuda.grid(1)
        if i > spinTrajecotires.shape[0]:
            return
        
        """"
        Simulate molecular diffusion of spins within the cells; dirichelt boundary conditions

        Parameters
        ----------
        rng_states : 1-d ndarray
            The random states of the spins
        spinTrajectories :  N_{fiber_spins} x 3 ndarray
            The spinTrajectories[i,:] sub-array are the spins physical cordinates
        cellAtSpin_i: 1-d ndarray
            The index of the cell a spin is located within; -1 if False, 0... N_fibers if True.
        numSteps: int
            int(Delta/dt); the number of timesteps
        cellCenters: N_{cells} x 4 ndarray
            The 3-spatial dimensions and radius of the i-th cell
        fiberCenters: N_{fibers} x 6 ndarray
            The spatial position, rotmat index, intrinsic diffusivity, and radius of the i-th fiber
        fiberRotationReference: 2 x 3 ndarray
            The Ry(Theta_{i}).dot([0,0,1]) vector  
        dt : float32
            The time-discretization parameter

        Returns
        -------
        spinTrajectories: N_{fiber_spins} x 3 ndarray  
            for 0 <= step <= numSteps, the spins trajectory is continuously updated until the spin steps within the cell but not within the fiber.           

        Notes
        -----
        re-stepping (currently implemented) vs. stepping is an ongoing discussion between me and S.K. Song about what is more reasonable. Tentatively, re-stepping allows for 
        accurate simulation at lower temporal resolutions, which gives favorable preformance results at the cost of some physical realism. 
        
        References
        ----------
        Discussions with S.K. Song about what physics is reasonable to implement here. 

        """
        
        inx = int32(cellAtSpin_i[i])
        D = float32(2.0)
        Step = float32(math.sqrt(6*D*dt))
        prevPosition = cuda.local.array(shape = 3, dtype= float32)
        newPosition = cuda.local.array(shape = 3, dtype= float32)
        distanceCell = float32(0.0)
        distanceFiber = float32(0.0)
        for step in range(numSteps):
            invalidMove = True
            while(invalidMove):
                isInFiber = False
                isNotInCell = False 
                newPosition = jp.randomDirection(rng_states, newPosition, i)
                for k in range(newPosition.shape[0]):
                    prevPosition[k] = spinTrajecotires[i,k]
                    newPosition[k] = prevPosition[k] + Step*newPosition[k]
                distanceCell = jp.euclidean_distance(newPosition, cellCenters[inx,0:3], fiberRotationReference[0,:], 'cell')
                if distanceCell > cellCenters[inx,3]:
                    isNotInCell = True
                    for k in range(newPosition.shape[0]): newPosition[k] = prevPosition[k]
                else:
                    for j in range(fiberCenters.shape[0]):
                        rotation_Index = int(fiberCenters[j,4])
                        distanceFiber = jp.euclidean_distance(newPosition, fiberCenters[j,0:3], fiberRotationReference[rotation_Index,:], 'fiber')
                        if distanceFiber < fiberCenters[j,3]:
                            isInFiber = True
                            for k in range(newPosition.shape[0]): newPosition[k] = prevPosition[k]
                            break
                if (not isInFiber) & (not isNotInCell):
                    invalidMove = False
                else:
                    invalidMove = True

            cuda.syncthreads()
            for k in range(newPosition.shape[0]): spinTrajecotires[i,k] = newPosition[k]
            cuda.syncthreads()  
        return 

    @cuda.jit 
    def diffusion_in_water(rng_states, spinTrajectories, numSteps, cellCenters, fiberCenters, fiberRotationReference, dt):
        i = cuda.grid(1)
        if i > spinTrajectories.shape[0]:
            return   
        """"
        Simulate molecular diffusion of spins within the extra-cellular and extra-axonal environment; dirichelt boundary conditions (i.e., spins in the extra-cellular/fiber environment
        are not allowed to diffuse into a fiber or into a cell). This step involves significant computation on O(N_{fibers}+N_{cells}). 

        Parameters
        ----------
        rng_states : 1-d ndarray
            The random states of the spins
        spinTrajectories :  N_{fiber_spins} x 3 ndarray
            The spinTrajectories[i,:] sub-array are the spins physical cordinates
        numSteps: int
            int(Delta/dt); the number of timesteps
        cellCenters: N_{cells} x 4 ndarray
            The 3-spatial dimensions and radius of the i-th cell
        fiberCenters: N_{fibers} x 6 ndarray
            The spatial position, rotmat index, intrinsic diffusivity, and radius of the i-th fiber
        fiberRotationReference: 2 x 3 ndarray
            The Ry(Theta_{i}).dot([0,0,1]) vector  
        dt : float32
            The time-discretization parameter

        Returns
        -------
        spinTrajectories: N_{fiber_spins} x 3 ndarray  
            for 0 <= step <= numSteps, the spins trajectory is continuously updated until the spin steps within the extra-cellular/extra-fiber environment.           

        Notes
        -----
        re-stepping (currently implemented) vs. stepping is an ongoing discussion between me and S.K. Song about what is more reasonable. Tentatively, re-stepping allows for 
        accurate simulation at lower temporal resolutions, which gives favorable preformance results at the cost of some physical realism. 
        
        References
        ----------
        Discussions with S.K. Song about what physics is reasonable to implement here. 

        """
        D = float32(3.0)
        Step = float32(math.sqrt(6*D*dt))
        prevPosition = cuda.local.array(shape = 3, dtype= float32)
        newPosition = cuda.local.array(shape = 3, dtype= float32)
        distanceCell = float32(0.0)
        distanceFiber = float32(0.0)
       
        for step in range(numSteps): 
            invalidStep = True
            while invalidStep:    
                inFiber = False
                inCell = False
                newPosition = jp.randomDirection(rng_states, newPosition, i)        
                for k in range(newPosition.shape[0]): 
                    prevPosition[k] = spinTrajectories[i,k]
                    newPosition[k] = prevPosition[k] + Step * newPosition[k]
                for l in range(cellCenters.shape[0]):
                    distanceCell = jp.euclidean_distance(newPosition, cellCenters[l, 0:3], fiberRotationReference[0,:], 'cell')
                    if distanceCell < cellCenters[l,3]:
                        inCell = True
                        #for k in range(newPosition.shape[0]): newPosition[k] = prevPosition[k]
                        break
                for l in range(fiberCenters.shape[0]):
                    rotationIndex = int(fiberCenters[l,4])
                    distanceFiber = jp.euclidean_distance(newPosition, fiberCenters[l,0:3], fiberRotationReference[rotationIndex,:], 'fiber')
                    if distanceFiber < fiberCenters[l,3]:
                        inFiber = True
                        #for k in range(newPosition.shape[0]): newPosition[k] = prevPosition[k]
                        break
                if (not inFiber):
                    invalidStep = False
                else:
                    invalidStep = True
            
            cuda.syncthreads()
            for k in range(newPosition.shape[0]): spinTrajectories[i,k] = newPosition[k]
            cuda.syncthreads()  
        return 
    
    def spins_in_voxel(self, trajectoryT1m, trajectoryT2p):
        """
         Helper function to ensure that the spins at time T2p are wtihin the self.voxelDims x self.voxelDims x inf imaging voxel

        Parameters
        ----------
        trajectoryT1m: N_{spins} x 3 ndarray
            The initial spin position at time t1m
        
        trajectoryT2p: N_{spins} x 3 ndarray
            The spin position at time t2p

        Returns
        -------
        traj1_vox: (N, 3) ndarray
            Position at T1m of the spins which stay within the voxel
        traj2_vox: (N, 3) ndarray
            Position at T2p of the spins which stay within the voxel

        Notes
        -----
        None
        
        References
        ----------
        None
        
        """
    
        traj1_vox = []
        traj2_vox = []

        for i in range(trajectoryT1m.shape[0]):
            if np.amin(trajectoryT2p[i,0:2]) >= 0 + 0.5*self.buffer and np.amax(trajectoryT2p[i,0:2]) <= self.voxelDims + 0.5*self.buffer:
                traj1_vox.append(trajectoryT1m[i,:])
                traj2_vox.append(trajectoryT2p[i,:])
        return np.array(traj1_vox), np.array(traj2_vox) 

    def signal(self, trajectoryT1m, trajectoryT2p, xyz):
        """
        Aquire the signal by integrating the ensemble distribution from t1m to t2p; int

        Parameters
        ----------
        trajectoryT1m: N_{spins} x 3 ndarray
            The initial spin position at time t1m
        
        trajectoryT2p: N_{spins} x 3 ndarray
            The spin position at time t2p

        Returns
        -------
        allSignal: (N_{bvals}, ) ndarray
            The signal induced by the k-th diffusion gradient and diffusion weighting factor
        
        b_vals: (N_{bvals},) ndarray
            The b-values used in the diffusion experiment

        Notes
        -----
        None
        
        References
        ----------
        [1] ... Rafael-Patino et. al. (2020)  Robust Monte-Carlo Simulations in Diffusion-MRI: 
                Effect of the Substrate Complexity and Parameter Choice on the Reproducibility of Results, 
                Front. Neuroinform., 10 March 2020 
        
        """
        
        gamma = 42.58
        Delta = self.Delta #ms
        dt = self.delta # ms 
        delta = dt #ms
        b_vals = np.linspace(0, 2200, 20)
        trajectoryT1m, trajectoryT2p = self.spins_in_voxel(trajectoryT1m, trajectoryT2p)
        if xyz:
            Gt = np.sqrt(10**-3 * b_vals/(gamma**2 * delta**2*(Delta-delta/3)))
            unitGradients = np.zeros((3*len(Gt), 3))
            for i in (range(unitGradients.shape[1])):
                unitGradients[i*len(b_vals): (i+1)*len(b_vals),i] = Gt
        else:
            unitGradients = self.bvecs.T
            b_vals = self.bvals
        allSignal = np.zeros(unitGradients.shape[0])
        for i in tqdm(range(unitGradients.shape[0])):
            signal = 0
            if xyz:
                scaledGradient = unitGradients[i,:]
            else:
                scaledGradient = np.sqrt( (b_vals[i] * 10**-3)/ (gamma**2*delta**2*(Delta - delta/3))) * unitGradients[i,:]
            for j in range(trajectoryT1m.shape[0]):
                phase_shift = gamma * np.sum(scaledGradient.dot(trajectoryT1m[j,:]-trajectoryT2p[j,:])) * dt
                signal = signal + np.exp(-1 *(0+1j) * phase_shift)
            signal = signal/trajectoryT1m.shape[0]
            allSignal[i] = np.abs(signal)
        dwi = nb.Nifti1Image(allSignal.reshape(1,1,1,-1), affine = np.eye(4))
        return allSignal, b_vals
    
    def add_noise(self, signal, snr, noise_type):
        sigma = 1.0/snr 
        real_channel_noise = np.random.normal(0, sigma, signal.shape[0])
        complex_channel_noise = np.random.normal(0, sigma, signal.shape[0])
        if noise_type == 'Rician':
            return np.sqrt((signal + real_channel_noise)**2 + complex_channel_noise**2)
        if noise_type == 'Gaussian':
            return signal + real_channel_noise

    def save_data(self, path, plot_xyz):

        data_dir = path + os.sep + "FF={}_Theta={}_Diffusions={}_Simulation".format(self.fiberFraction, self.Thetas, self.fiberDiffusions)
        if not os.path.exists(data_dir): os.mkdir(data_dir)
        overallData = []
        if self.simulateFibers:
            fiber_trajectories = [self.fiberPositionsT1m, self.fiberPositionsT2p]
            overallData.append(fiber_trajectories)
            print(self.fiberPositionsT1m.shape[0])
            np.save(data_dir + os.sep + "fiberPositionsT1m_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.fiberPositionsT1m)
            np.save(data_dir + os.sep + "fiberPositionsT2p_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.fiberPositionsT2p)
            pureFiberSignal, b = self.signal(self.fiberPositionsT1m, self.fiberPositionsT2p, xyz = False)
            dwiFiber = nb.Nifti1Image(pureFiberSignal.reshape(1,1,1,-1), affine = np.eye(4))
            nb.save(dwiFiber, data_dir + os.sep + "pureFiberSignal_angle={}_diffusivities={}_dt={}_ff={}.nii".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)))

        if self.simulateCells:
            cell_trajectories = [self.cellPositionsT1m, self.cellPositionsT2p]
            overallData.append(cell_trajectories)
            print(self.cellPositionsT1m.shape[0])
            np.save(data_dir + os.sep + "cellPositionsT1m_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.cellPositionsT1m)
            np.save(data_dir + os.sep + "cellPositionsT2p_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.cellPositionsT2p)
            pureCellSignal, _ = self.signal(self.cellPositionsT1m, self.cellPositionsT2p, xyz = True)
            dwiCell = nb.Nifti1Image(pureCellSignal.reshape(1,1,1,-1), affine = np.eye(4))
            nb.save(dwiCell, data_dir + os.sep + "pureCellSignal_angle={}_diffusivities={}_dt={}_ff={}.nii".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)))
        if self.simulateExtra:
            water_trajectories = [self.extraPositionT1m, self.extraPositionT2p]
            overallData.append(water_trajectories)
            print(self.extraPositionT1m.shape[0])
            np.save(data_dir + os.sep + "waterPositionsT1m_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.extraPositionT1m)
            np.save(data_dir + os.sep + "waterPositionsT2p_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.extraPositionT2p)
            pureWaterSignal, _ = self.signal(self.extraPositionT1m, self.extraPositionT2p, xyz = True)
            dwiWater = nb.Nifti1Image(pureWaterSignal.reshape(1,1,1,-1), affine = np.eye(4))
            nb.save(dwiWater, data_dir + os.sep  + "pureWaterSignal_angle={}_diffusivities={}_dt={}_ff={}.nii".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)))
        
        if self.simulateFibers and self.simulateExtra:
            expSignal, bvals = self.signal(np.vstack([self.fiberPositionsT1m, self.extraPositionT1m]), np.vstack([self.fiberPositionsT2p, self.extraPositionT2p]), xyz = True)
            dwi = nb.Nifti1Image(expSignal.reshape(1,1,1,-1), affine = np.eye(4))
            nb.save(dwi,data_dir + os.sep + "totalSignal_angle={}_diffusivities={}_dt={}_ff={}.nii".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)))

    def plot(self, plotFibers, plotCells, plotExtra):
        def plot_fiber(self, fiberCenter):
            x_ctr = fiberCenter[0]
            y_ctr = fiberCenter[1]
            radius = fiberCenter[3]
            z = np.linspace(-10, 200+10,2)
            theta = np.linspace(0,2*np.pi,25)
            th, zs = np.meshgrid(theta, z)
            xs = (radius * np.cos(th) + x_ctr)
            ys = (radius * np.sin(th) + y_ctr)
            
            if fiberCenter[4] == 0:
                return xs,ys,zs
            else:
                rotation = np.dot(self.rotMat, np.array([xs.ravel(), ys.ravel(), zs.ravel()]))
                x = rotation[0,:].reshape(xs.shape) 
                y = rotation[1,:].reshape(ys.shape)
                z = rotation[2,:].reshape(zs.shape) + np.abs(np.amin(rotation[2,:]))
                print(np.abs(np.amin(rotation[2,:])))
                return x,y,z
        
        fig = plt.figure(figsize = (8,8))

        if plotFibers and self.simulateFibers:

            fiber_indicies = (self.spinInFiber_i[self.spinInFiber_i > 0]).astype(np.int16)

            bundle_1 = self.fiberPositionsT2p[[i for i in range(len(fiber_indicies)) if self.fiberCenters[fiber_indicies[i],4] == 1.0],:]
            bundle_2 = self.fiberPositionsT2p[[i for i in range(len(fiber_indicies)) if self.fiberCenters[fiber_indicies[i],4] == 0.0],:]


        


            axFiber = fig.add_subplot(projection = '3d')
            axFiber.scatter(bundle_1[:,0], bundle_1[:,1], bundle_1[:,2], s = 2, color = 'mediumseagreen', label = 'Bundle 1', alpha = .50)
            axFiber.scatter(bundle_2[:,0], bundle_2[:,1], bundle_2[:,2], s = 2, label = 'Bundle 2')
            #axFiber.view_init(elev=90, azim=0)
            #axFiber.scatter(self.fiberCenters[:,0], self.fiberCenters[:,1], self.fiberCenters[:,2])
            axFiber.set_xlim(0,self.voxelDims+self.buffer)
            axFiber.set_ylim(0,self.voxelDims+self.buffer)
            axFiber.set_zlim(0,self.voxelDims+self.buffer)
            axFiber.set_xlabel('x')
            axFiber.set_ylabel('y')
            axFiber.set_zlabel('z')
            axFiber.legend(markerscale=10)
        
        if plotExtra and self.simulateExtra:
            axExtra = fig.add_subplot(projection = '3d')
            axExtra.scatter(self.extraPositionT2p[:,0], self.extraPositionT2p[:,1], self.extraPositionT2p[:,2], s = 1)
            axExtra.set_xlim(0,self.voxelDims+self.buffer)
            axExtra.set_ylim(0,self.voxelDims+self.buffer)
            axExtra.set_zlim(0,self.voxelDims+self.buffer)
            axExtra.set_xlabel(r'x $\quad \mu m$')
            axExtra.set_ylabel(r'y $\quad \mu m$')
            axExtra.set_zlabel(r'z $\quad \mu m$')
            axExtra.legend()

        if any([plotFibers, plotExtra, plotCells]):
            plt.show()


def dmri_sim_wraper(arg):
    x = dmri_simulation()
    x.from_config(arg)

def main():       
    configs = glob.glob(r"C:\MCSIM\dMRI-MCSIM-main\run_from_config_test\IW\simulation_configuration_Theta=*_Fraction=*.ini")
    for cfg in configs:
        p = Process(target=dmri_sim_wraper, args = (cfg,))
        p.start()
        p.join()

    exit()

    sim.set_parameters(
        numSpins= 100*10**3,
        fiberFraction= (0.50, .50),  # Fraction in each Half/Quadrant Depending on 'P'/'NP'
        fiberRadius= 1.0,            # um
        Thetas = (0,0),              # degrees
        fiberDiffusions= (1.0, 1.0), # um^2/mm
        cellFraction= .0,            # Fraction in each Half/Quadrant Depending on 'P'/'NP'
        cellRadii= (3,10),           # um
        fiberConfiguration = 'Inter-Woven',           # 'P' = Penetrating Cells; 'NP = Non-Penetrating Cells, 'IW' 
        Delta = 10,                  # ms  
        dt = 0.001,                  # ms 
        voxelDim= 20,                # um
        buffer = 10,                 # um
        path_to_bvals= r"C:\MCSIM\Repo\simulation_data\DBSI\DBSI-99\bval",
        path_to_bvecs= r"C:\MCSIM\Repo\simulation_data\DBSI\DBSI-99\bvec"
        )   
    grid = sim.fiberCenters
    
    
    print('numSpins: {}'.format(sim.numSpins))
    print('fiberFractions: {}'.format(sim.fiberFraction))
    print('Inter Fiber Distance: {}'.format(np.linalg.norm(grid[0,0:3]-grid[1,0:3], ord = 2) - 2*sim.fiberRadius))
    cont = input(r"Do you wish to proceed with these parameters: [y\n] ")



    if cont == 'y':

        sim.simulate(simulateFibers=True, simulateCells=False, simulateExtraEnvironment=False)
        sim.plot(plotFibers=True, plotCells=False, plotExtra=False)
        
        
        exit()
        
        sim.save_data(r"C:\MCSIM\dMRI-MCSIM-main\gpu_data")
        s, b = sim.signal(sim.fiberPositionsT1m, sim.fiberPositionsT2p, xyz = True)
        ifd = np.linalg.norm(grid[0,0:3]-grid[1,0:3], ord = 2) - 2*sim.fiberRadius
        ('Inter Fiber Distance: {}'.format(ifd))

        A = np.ones((b.shape[0],2))
        A[:,1] = -1/1000 * b
        bx, mx = np.linalg.lstsq(A, np.log(s[0:20]), rcond=None)[0]
        by, my = np.linalg.lstsq(A, np.log(s[20:40]), rcond=None)[0]
        bz, mz = np.linalg.lstsq(A, np.log(s[40:60]), rcond=None)[0]
        plt.plot(b, (np.log(s[0:20])), 'bx', label = mx)
        plt.plot(b, np.log(s[20:40]), 'rx', label = my)
        plt.plot(b, np.log(s[40:60]), 'gx', label = mz)
        plt.legend()
        plt.savefig(r"C:\MCSIM\dMRI-MCSIM-main\gpu_data\fiber_signal_angle={}_diffusivities={}_dt={}_ff={}_dist={}.png".format(sim.Thetas, sim.fiberDiffusions,sim.dt,sim.fiberFraction, round(ifd,4)))

        plt.clf()

        #sWater, bWater = sim.signal(sim.extraPositionT1m, sim.extraPositionT2p, xyz = True)
       # 
       # bwx, mwx = np.linalg.lstsq(A, np.log(sWater[0:20]), rcond=None)[0]
       # bwy, mwy = np.linalg.lstsq(A, np.log(sWater[20:40]), rcond=None)[0]
       # bwz, mwz = np.linalg.lstsq(A, np.log(sWater[40:60]), rcond=None)[0]
       # plt.plot(bWater, (sWater[0:20]), 'bx', label = mwx)
       # plt.plot(bWater, (sWater[20:40]), 'rx', label = mwy)
       # plt.plot(bWater, (sWater[40:60]), 'gx', label = mwz)
       # plt.legend()
       # plt.savefig(r"C:\MCSIM\dMRI-MCSIM-main\gpu_data\extra_signal_angle={}_diffusivities={}_dt={}_ff={}_dist={}.png".format(sim.Thetas, sim.fiberDiffusions,sim.dt,sim.fiberFraction, round(ifd,4)))

        #plt.clf()
        End = time.time()
        sim.plot(plotFibers=True, plotCells=False,plotExtra=False)
        print('Simulation Executed in: {} sec'.format(End-Start))
    return
   



if __name__ == "__main__":
    main()

    
 



