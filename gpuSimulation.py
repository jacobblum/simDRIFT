from cProfile import label
from pickletools import read_unicodestring1
import numpy as np 
import numba 
from numba import jit, cuda, int32, float32
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
import shutil
import walk_in_fiber
import walk_in_cell
import walk_in_extra_environ
import sys


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
        voidDist = 0
        buffer = 100
        bvals = 0
        bvecs = 0
        cfg_path = ''
        path_to_save = ''

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
        self.voidDist = .60*self.voxelDims
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

    def _set_params_from_config(self, path_to_configuration_file):
        self.cfg_path = path_to_configuration_file
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
        self.simulateFibers = literal_eval(config['Simulation Parameters']['simulateFibers'])
        self.simulateCells = literal_eval(config['Simulation Parameters']['simulateCells'])
        self.simulateExtra = literal_eval(config['Simulation Parameters']['simulateExtraEnvironment'])

        ## Scanning Parameters
        Delta = literal_eval(config['Scanning Parameters']['Delta'])
        dt = literal_eval(config['Scanning Parameters']['dt'])
        voxelDims = literal_eval(config['Scanning Parameters']['voxelDim'])
        buffer = literal_eval(config['Scanning Parameters']['buffer'])
        bvals_path = config['Scanning Parameters']['path_to_bvals']
        bvecs_path = config['Scanning Parameters']['path_to_bvecs']

        ## Saving Parameters
        self.path_to_save = config['Saving Parameters']['path_to_save_file_dir']

        
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

        return

    def from_config(self, path_to_configuration_file):
        self._set_params_from_config(path_to_configuration_file)
        
        

        rng_states_gpu = cuda.to_device(create_xoroshiro128p_states(self.spinPotionsT1m.shape[0], seed = 42))
        spin_positions_t1m = self.spinPotionsT1m.copy()
        spin_positions_gpu = cuda.to_device(self.spinPotionsT1m)
        spin_in_fiber_at_index_gpu = cuda.to_device(self.spinInFiber_i)
        fiber_centers_gpu = cuda.to_device(self.fiberCenters)
        spin_in_cell_at_index_gpu = cuda.to_device(self.spinInCell_i)
        cell_centers_gpu = cuda.to_device(self.cellCenters)
        rotation_reference_gpu = cuda.to_device(self.fiberRotationReference)
        dt_gpu = (self.dt)

        N_iter = int(self.Delta/self.dt)
        
        for i in (range(N_iter)):
            sys.stdout.write('\r' + str(i+1) + '/' + str(N_iter))
            sys.stdout.flush()
            cuda.synchronize()
            self._diffusion_context_manager.forall(spin_positions_gpu.shape[0])(
                rng_states_gpu,
                spin_positions_gpu,
                spin_in_fiber_at_index_gpu,
                fiber_centers_gpu,
                spin_in_cell_at_index_gpu,
                cell_centers_gpu,
                rotation_reference_gpu,
                dt_gpu,
                self.fiberCofiguration == 'Void'
            )
            cuda.synchronize()
        print('\n')
        print('Simulation executed')
        



        spin_positions_t2p = spin_positions_gpu.copy_to_host()
        
        print('Coppied to Host')
        
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(projection = '3d')
        ax.scatter(self.fiberCenters[:,0], self.fiberCenters[:,1], self.fiberCenters[:,2])
        ax.scatter(spin_positions_t2p[np.where(self.spinInFiber_i == -1)][:,0],spin_positions_t2p[np.where(self.spinInFiber_i == -1)][:,1], spin_positions_t2p[np.where(self.spinInFiber_i == -1)][:,2], s = 1)
        plt.show()
       
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

        if self.fiberCofiguration == 'Penetrating' or self.fiberCofiguration == 'Void':
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
        
        if self.fiberCofiguration == 'Void':
            null_index = np.where((outputArg[:,1] > 0.5*(self.voxelDims+self.buffer)-0.5*self.voidDist) & 
                                    (outputArg[:,1] < 0.5*(self.voxelDims+self.buffer)+0.5*self.voidDist))
            outputArg[null_index] = 0
        final = outputArg[~np.all(outputArg == 0, axis=1)]
        return final
    
    def place_cell_grid(self):
        print('PLACING CELLS')
        numCells = self.numCells
        cellCentersTotal = []

        if self.fiberCofiguration == 'Non-Penetrating':
            regions = np.array([[0,self.voxelDims+self.buffer,0,0.5*(self.voxelDims+self.buffer),0.5*(self.voxelDims+self.buffer), self.voxelDims+self.buffer], 
                                [0,0.5*(self.voxelDims+self.buffer),0.5*(self.voxelDims+self.buffer), self.voxelDims+self.buffer,0,self.voxelDims+self.buffer]])
        
        elif self.fiberCofiguration == 'Void':
            regions = np.array([[0, self.voxelDims+self.buffer, 0.5*(self.voxelDims+self.buffer)-0.5*self.voidDist, 0.5*(self.voxelDims+self.buffer)+0.5*(self.voidDist), 0, self.voxelDims+self.buffer], 
                                [0, self.voxelDims+self.buffer, 0.5*(self.voxelDims+self.buffer)-0.5*self.voidDist, 0.5*(self.voxelDims+self.buffer)+0.5*self.voidDist,0,self.voxelDims+self.buffer]])            
        else: 
            regions = np.array([[0,self.voxelDims+self.buffer,0,0.5*(self.voxelDims+self.buffer), 0,self.voxelDims+self.buffer],
                                [0,self.voxelDims+self.buffer,0.5*(self.voxelDims+self.buffer),self.voxelDims+self.buffer,0,self.voxelDims+self.buffer]])
        
        for i in (range(len(numCells))):
            print('{} / {}'.format(i+1, len(numCells)))
            cellCenters = np.zeros((numCells[i]**3, 4))
            for j in range(cellCenters.shape[0]):
                if j == 0:
                    invalid = True 
                    while(invalid):   
                        radius = self.cellRadii[i]
                        xllim, xulim = regions[i,0], regions[i,1]
                        yllim, yulim = regions[i,2], regions[i,3]
                        zllim, zulim = regions[i,4], regions[i,5]
                        cell_x = np.random.uniform(xllim + radius, xulim - radius)
                        cell_y = np.random.uniform(yllim + radius, yulim - radius)
                        cell_z = np.random.uniform(zllim + radius, zulim - radius)
                        cell_0 = np.array([cell_x, cell_y, cell_z, radius])
                        propostedCell = cell_0
                        ctr = 0
                        if i == 0:
                            cellCenters[j,:] = propostedCell
                            invalid = False
                        elif i > 0:
                            for k in range(cellCentersTotal[0].shape[0]):
                                distance = np.linalg.norm(propostedCell-cellCentersTotal[0][k,:], ord = 2)
                                if distance < (radius + cellCentersTotal[0][k,3]):
                                        ctr += 1
                                        break
                        if ctr == 0:
                            cellCenters[j,:] = propostedCell
                            invalid = False
                elif (j > 0):
                    invalid = True
                    while(invalid):
                        xllim, xulim = regions[i,0], regions[i,1]
                        yllim, yulim = regions[i,2], regions[i,3]
                        zllim, zulim = regions[i,4], regions[i,5]
                        radius = self.cellRadii[i]
                        cell_x = np.random.uniform(xllim + radius, xulim - radius)
                        cell_y = np.random.uniform(yllim + radius, yulim - radius)
                        cell_z = np.random.uniform(zllim + radius, zulim - radius)
                        propostedCell = np.array([cell_x, cell_y, cell_z, radius])
                        ctr = 0
                        for k in range(j):
                            distance = np.linalg.norm(propostedCell-cellCenters[k,:], ord = 2)
                            if distance < 2*radius:
                                ctr += 1
                                break
                            if i > 0:
                                for l in range(cellCentersTotal[0].shape[0]):
                                    distance = np.linalg.norm(propostedCell-cellCentersTotal[0][l,:], ord = 2)
                                    if distance < (radius + cellCentersTotal[0][l,3]):
                                        ctr += 1
                                        break
                        if ctr == 0:
                            cellCenters[j,:] = propostedCell
                            invalid = False
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
            extraSpins_GPU = cuda.to_device(self.extraPositionT1m.astype(np.float32).copy())
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



    @numba.cuda.jit
    def _diffusion_context_manager(rng_states, spin_positions, spin_in_fiber_key, fiber_centers, spin_in_cell_key, cell_centers, fiber_rotation_reference, dt, void):
        """
        Parameters:

        rng_states:
        spinPositions: (N_spins, 3) numpy array
        spin_in_fiber_key: (N_spins, ) numpy array, the spin_in_fiber[i] is the fiber index of the i-th spin. -1 if spin not in fiber.
        
        
        """
        gpu_index = cuda.grid(1)
        if gpu_index > spin_positions.shape[0]:
            return
        
        spin_in_fiber_index = int(spin_in_fiber_key[gpu_index])
        spin_in_cell_index  = int(spin_in_cell_key[gpu_index])

        spin_in_fiber_boolean = (spin_in_fiber_index > -1)
        spin_in_cell_boolean  = ((spin_in_cell_index > -1) & (spin_in_fiber_index == -1)) 

        if spin_in_fiber_boolean:
            walk_in_fiber._diffusion_in_fiber(gpu_index, rng_states, spin_in_fiber_index, spin_positions, fiber_centers, fiber_rotation_reference, dt)
        
        if spin_in_cell_boolean:
            walk_in_cell._diffusion_in_cell(gpu_index, rng_states, spin_positions, spin_in_cell_index, cell_centers, fiber_centers, fiber_rotation_reference, dt, void)
        
        if (not(spin_in_cell_boolean)) & (not(spin_in_fiber_boolean)):
            walk_in_extra_environ._diffusion_in_extra_environment(gpu_index,rng_states,spin_positions,fiber_centers, cell_centers, fiber_rotation_reference, dt)
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
            for k in range(newPosition.shape[0]):
                spinTrajectories[i,k] = newPosition[k]
            if i == 0 and (step % (numSteps / 5) == 0):
                print('Fiber Step: ', step,' (',int((step/numSteps)*100.),' %)')
            cuda.syncthreads()
        return

    @cuda.jit
    def diffusion_in_cell(rng_states, spinTrajectories, cellAtSpin_i, numSteps, cellCenters, fiberCenters, fiberRotationReference, dt):
        i = cuda.grid(1)
        if i > spinTrajectories.shape[0]:
            return
        
        """"
        Simulate molecular diffusion of spins within the cells; Dirichelt boundary conditions

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
                    prevPosition[k] = spinTrajectories[i,k]
                    newPosition[k] = prevPosition[k] + Step*newPosition[k]
                distanceCell = jp.euclidean_distance(newPosition, cellCenters[inx,0:3], fiberRotationReference[0,:], 'cell')
                if distanceCell > cellCenters[inx,3]:
                    isNotInCell = True
                   # for k in range(newPosition.shape[0]): newPosition[k] = prevPosition[k]
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
            for k in range(newPosition.shape[0]):
                spinTrajectories[i,k] = newPosition[k]
            if i == 0 and (step % (numSteps / 5) == 0):
                print('Cell Step: ', step,' (',int((step/numSteps)*100.),' %)')
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
            for k in range(newPosition.shape[0]): 
                spinTrajectories[i,k] = newPosition[k]
            if i == 0 and (step % (numSteps / 5) == 0):
                print('Water Step: ', step,' (',int((step/numSteps)*100.),' %)')
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

    def signal(self, trajectoryT1m, trajectoryT2p, xyz, finite):
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
        if finite: trajectoryT1m, trajectoryT2p = self.spins_in_voxel(trajectoryT1m, trajectoryT2p)
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
        data_dir = path + os.sep + "FF={}_CF={}_CellRad={}_Theta={}_Diffusions={}_fibConfig={}_Sim".format(self.fiberFraction, self.cellFraction, self.cellRadii, self.Thetas, self.fiberDiffusions, self.fiberCofiguration)
        if not os.path.exists(data_dir): os.mkdir(data_dir)
        path, file = os.path.split(self.cfg_path)  
        if not os.path.exists(data_dir + os.sep + file): shutil.move(self.cfg_path, data_dir + os.sep + file)
        overallData = []
        if self.simulateFibers:
            fiber_trajectories = [self.fiberPositionsT1m, self.fiberPositionsT2p]
            overallData.append(fiber_trajectories)
            print(self.fiberPositionsT1m.shape[0])
            np.save(data_dir + os.sep + "fiberPositionsT1m_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.fiberPositionsT1m)
            np.save(data_dir + os.sep + "fiberPositionsT2p_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.fiberPositionsT2p)
            pureFiberSignal, b = self.signal(self.fiberPositionsT1m, self.fiberPositionsT2p, xyz = False, finite = False)
            dwiFiber = nb.Nifti1Image(pureFiberSignal.reshape(1,1,1,-1), affine = np.eye(4))
            nb.save(dwiFiber, data_dir + os.sep + "pureFiberSignal_angle={}_diffusivities={}_dt={}_ff={}.nii".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)))

        if self.simulateCells:
            cell_trajectories = [self.cellPositionsT1m, self.cellPositionsT2p]
            overallData.append(cell_trajectories)
            print(self.cellPositionsT1m.shape[0])
            np.save(data_dir + os.sep + "cellPositionsT1m_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.cellPositionsT1m)
            np.save(data_dir + os.sep + "cellPositionsT2p_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.cellPositionsT2p)
            pureCellSignal, _ = self.signal(self.cellPositionsT1m, self.cellPositionsT2p, xyz = False, finite=False)
            dwiCell = nb.Nifti1Image(pureCellSignal.reshape(1,1,1,-1), affine = np.eye(4))
            nb.save(dwiCell, data_dir + os.sep + "pureCellSignal_angle={}_diffusivities={}_dt={}_ff={}.nii".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)))
        if self.simulateExtra:
            water_trajectories = [self.extraPositionT1m, self.extraPositionT2p]
            overallData.append(water_trajectories)
            print(self.extraPositionT1m.shape[0])
            np.save(data_dir + os.sep + "waterPositionsT1m_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.extraPositionT1m)
            np.save(data_dir + os.sep + "waterPositionsT2p_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.extraPositionT2p)
            pureWaterSignal, bvals = self.signal(self.extraPositionT1m, self.extraPositionT2p, xyz = False,finite=False)
            dwiWater = nb.Nifti1Image(pureWaterSignal.reshape(1,1,1,-1), affine = np.eye(4))
            nb.save(dwiWater, data_dir + os.sep  + "pureWaterSignal_angle={}_diffusivities={}_dt={}_ff={}.nii".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)))

        if self.simulateFibers and self.simulateExtra and self.simulateCells:
            expSignal, bvals = self.signal(np.vstack([self.fiberPositionsT1m, self.cellPositionsT1m,self.extraPositionT1m]), np.vstack([self.fiberPositionsT2p,self.cellPositionsT2p,self.extraPositionT2p]), xyz = False, finite = False)
            dwi = nb.Nifti1Image(expSignal.reshape(1,1,1,-1), affine = np.eye(4))
            nb.save(dwi,data_dir + os.sep + "totalSignal_angle={}_diffusivities={}_dt={}_ff={}.nii".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)))

    def plot(self, plotFibers, plotCells, plotExtra, plotConfig):
        def plot_fiber(self, fiberCenters):

            translation_factor = np.zeros(fiberCenters.shape[0])
            vertResolution = 2
            rotResolution = 25

            z_min = 0

            Xs_b1 = []
            Ys_b1 = []
            Zs_b1 = []
            Xs_b2 = []
            Ys_b2 = []
            Zs_b2 = []
            b1 = []
            b2 = []

        
            for i in range(fiberCenters.shape[0]):
                fiberCenter = fiberCenters[i,:]
                x_ctr = fiberCenter[0]
                y_ctr = fiberCenter[1]
                radius = fiberCenter[3]
                z = np.linspace(z_min, (self.voxelDims + self.buffer), vertResolution)
                theta = np.linspace(0,2*np.pi,rotResolution)
                th, zs = np.meshgrid(theta, z)
                xs = (radius * np.cos(th) + x_ctr)
                ys = (radius * np.sin(th) + y_ctr)

                if fiberCenter[4] == 0:
                    Xs_b1.append(xs)
                    Ys_b1.append(ys)
                    Zs_b1.append(zs)
                    b1.append(fiberCenter[4])
                else:
                    rotation = np.dot(self.rotMat, np.array([xs.ravel(), ys.ravel(), zs.ravel()]))
                    translation_factor[i] = np.amin(rotation[2,:])
                    Xs_b2.append(rotation[0,:].reshape(xs.shape)) 
                    Ys_b2.append(rotation[1,:].reshape(xs.shape))
                    Zs_b2.append(rotation[2,:].reshape(xs.shape))
                    b2.append(fiberCenter[4])
            
            Xs_b1 = np.array(Xs_b1)
            Ys_b1 = np.array(Ys_b1)
            Zs_b1 = np.array(Zs_b1)

            Xs_b2 = np.array(Xs_b2)
            Ys_b2 = np.array(Ys_b2)
            
            Zs_b2 = np.array(Zs_b2) + np.abs(np.amin(translation_factor-z_min))
            return np.vstack([Xs_b1,Xs_b2]), np.vstack([Ys_b1, Ys_b2]), np.vstack([Zs_b1,Zs_b2]), np.array(np.hstack([b1, b2]))

        
        fig = plt.figure(figsize = (8,8))

        if plotConfig:

            axFibers = fig.add_subplot(projection = '3d')
           
            X,Y,Z, B = plot_fiber(self, fiberCenters = self.fiberCenters)
            for i in range(X.shape[0]):
                Xp, Yp, Zp = X[i,:,:], Y[i,:,:], Z[i,:,:]
                if B[i] == 1:       
                    surf = axFibers.plot_surface(Xp,Yp,Zp, color = 'r')
                else:
                    surf = axFibers.plot_surface(Xp,Yp,Zp, color = 'b')

            #axFibers.set_ylim(0.5*(self.buffer), self.voxelDims+0.5*self.buffer)
            #axFibers.set_xlim(0.5*(self.buffer), self.voxelDims+0.5*self.buffer)
            #axFibers.set_zlim(0.5*(self.buffer), self.voxelDims+0.5*self.buffer)
            plt.show()

        axFiber = fig.add_subplot(projection = '3d')

        if plotFibers:



            """

            fiber_indicies = (self.spinInFiber_i[self.spinInFiber_i > 0]).astype(np.int16)
            bundle_1_T1m = (self.fiberPositionsT1m[[i for i in range(len(fiber_indicies)) if self.fiberCenters[fiber_indicies[i],4] == 1.0],:])
            bundle_1_T2p = (self.fiberPositionsT2p[[i for i in range(len(fiber_indicies)) if self.fiberCenters[fiber_indicies[i],4] == 1.0],:])
            bundle_2_T1m = (self.fiberPositionsT1m[[i for i in range(len(fiber_indicies)) if self.fiberCenters[fiber_indicies[i],4] == 0.0],:])
            bundle_2_T2p = (self.fiberPositionsT2p[[i for i in range(len(fiber_indicies)) if self.fiberCenters[fiber_indicies[i],4] == 0.0],:])
            bundle_1_T1m, bundle_1_T2p = self.spins_in_voxel(bundle_1_T1m, bundle_1_T2p)
            bundle_2_T1m, bundle_2_T2p = self.spins_in_voxel(bundle_2_T1m, bundle_2_T2p)

            axFiber.scatter(bundle_1_T2p[:,0], bundle_1_T2p[:,1], bundle_1_T2p[:,2], s = 2, color = 'darkmagenta', label = 'Bundle 2')
            axFiber.scatter(bundle_2_T2p[:,0], bundle_2_T2p[:,1], bundle_2_T2p[:,2], s = 2, color = 'cadetblue', label = 'Bundle 1')
            #axFiber.view_init(elev=90, azim=0)
            #axFiber.scatter(self.fiberCenters[:,0], self.fiberCenters[:,1], self.fiberCenters[:,2])
            
            """

            axFiber.scatter(self.fiberCenters[:,0], self.fiberCenters[:,1], self.fiberCenters[:,2])
            
            axFiber.set_xlim(0.5*self.buffer,self.voxelDims+0.5*self.buffer)
            axFiber.set_ylim(0.5*self.buffer,self.voxelDims+0.5*self.buffer)
            axFiber.set_zlim(0.5*self.buffer,self.voxelDims+0.5*self.buffer)
            axFiber.set_xlabel(r'x $\quad \mu m$')
            axFiber.set_ylabel(r'y $\quad \mu m$')
            axFiber.set_zlabel(r'z $\quad \mu m$')
            axFiber.legend(markerscale=5)


        
        if plotCells:
            cell1 = self.cellCenters[np.where(self.cellCenters[:,3] == self.cellRadii[0])]
            cell2 = self.cellCenters[np.where(self.cellCenters[:,3] == self.cellRadii[1])]
            
            
            axFiber.scatter(cell1[:,0], cell1[:,1], cell1[:,2])
            axFiber.scatter(cell2[:,0], cell2[:,1], cell2[:,2])
        
        if plotExtra:
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

        return 

    def _signal_from_trajectory_data(self,trajectory_dir):
        trajectory_t1ms = glob.glob(trajectory_dir + os.sep + '*water*T1m*.npy')
        for trajectory_file in trajectory_t1ms:
                for f in ['fin', 'inf']:
                    traj_dir, fname = os.path.split(trajectory_file)
                    compartment = (fname[0:5])
                    traj1 = np.load(trajectory_file)
                    traj2 = np.load(trajectory_file.replace('T1m', 'T2p'))
                    fig, ax = plt.subplots(figsize = (10,3))
                    ax.hist(traj2-traj1, bins = 1000)
                    plt.show()
                    signal, bvals = self.signal(traj1, traj2, xyz = False, finite = (f == 'fin'))
                    plt.show()
                    dwi = nb.Nifti1Image(signal.reshape(1,1,1,-1), affine = np.eye(4))
                    nb.save(dwi, traj_dir + os.sep  + compartment + '_' + f + "_Signal_angle={}_diffusivities={}_dt={}_ff={}.nii".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)))
        return

def dmri_sim_wraper(arg):
    path, file = os.path.split(arg)
    x = dmri_simulation()
    x.from_config(arg)
    #x._set_params_from_config(arg)
    #x._signal_from_trajectory_data(path)
    

def main():       
    #numba.cuda.detect()
    configs = glob.glob(r"C:\MCSIM\dMRI-MCSIM-main\Yes_Cells\R4_config_Theta=(0, 90)_fibFrac=(0.1, 0.1)_cellFrac=0.35_cellRad=(2.5, 2.5)_Diff=(1.0, 2.0)_Pene.ini")
    for cfg in configs:
        p = Process(target=dmri_sim_wraper, args = (cfg,))
        p.start()
        p.join()
    
    """
    sim = dmri_simulation()
    sim.set_parameters(
        numSpins= 100*10**3,
        fiberFraction= (0.50, .50),  # Fraction in each Half/Quadrant Depending on 'P'/'NP'
        fiberRadius= 1.0,            # um
        Thetas = (0,90),              # degrees
        fiberDiffusions= (1.0, 1.0), # um^2/mm
        cellFraction= .10,            # Fraction in each Half/Quadrant Depending on 'P'/'NP'
        cellRadii= (10,5),           # um
        fiberConfiguration = 'Void',           # 'P' = Penetrating Cells; 'NP = Non-Penetrating Cells, 'IW' 
        Delta = 10,                  # ms  
        dt = 0.001,                  # ms 
        voxelDim= 50,                # um
        buffer = 0,                 # um
        path_to_bvals= r'C:\MCSIM\Repo\simulation_data\DBSI\DBSI-99\bval\bval-99.bval',
        path_to_bvecs= r'C:\MCSIM\Repo\simulation_data\DBSI\DBSI-99\bvec\bvec-99.bvec',
        path_to_save_file_dir= r'C:\MCSIM\dMRI-MCSIM-main\klu_test'
        )   
    """


if __name__ == "__main__":
    main()
    
 



