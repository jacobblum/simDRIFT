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
        penetrating = False
        buffer = 100
        bvals = 0
        bvecs = 0
        return
    
    def set_parameters(self, numSpins, fiberFraction, fiberRadius, Thetas, fiberDiffusions, cellFraction, cellRadii, penetrating, Delta, dt, voxelDim, buffer, path_to_bvals, path_to_bvecs):
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
        if penetrating == 'P':
            self.penetrating = True
        elif penetrating == 'NP':
            self.penetrating = False
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
            if not self.penetrating:
                fiberXs, fiberYs = np.meshgrid(np.linspace(start = (i*(self.voxelDims+self.buffer)*.50)+self.fiberRadius, stop = (i+1)*(self.voxelDims+self.buffer)*.50-self.fiberRadius, num = self.numFibers[i]), 
                                               np.linspace(start = (i*(self.voxelDims+self.buffer)*.50)+self.fiberRadius, stop = (i+1)*(self.voxelDims+self.buffer)*.50-self.fiberRadius, num = self.numFibers[i]))
            if self.penetrating:
                fiberXs, fiberYs = np.meshgrid(np.linspace(start = 0+self.fiberRadius, stop = self.voxelDims+self.buffer-self.fiberRadius, num = self.numFibers[i]), 
                                               np.linspace(start = (i*(self.voxelDims+self.buffer)*.50)+self.fiberRadius, stop = (i+1)*(self.voxelDims+self.buffer)*.50-self.fiberRadius, num = self.numFibers[i]))

            
            fiberXs, fiberYs = np.meshgrid(np.linspace(0+self.fiberRadius,self.voxelDims+self.buffer-self.fiberRadius, self.numFibers[0]), 
                                           np.linspace(0+self.fiberRadius,self.voxelDims+self.buffer-self.fiberRadius, self.numFibers[0]))
        
            fiberCordinates[:,0] = fiberXs.flatten()
            fiberCordinates[:,1] = fiberYs.flatten()
            fiberCordinates[:,2] = 1.0
            fiberCordinates[:,3] = self.fiberRadius
            fiberCordinates[fiberCordinates[:,1] < (self.voxelDims+self.buffer)/2, 4] = 1 # Assigns the rotation refernce index; 0 is for [0,0,1], 1 would correspond to Ry.dot([0,0,1])
            fiberCordinates[fiberCordinates[:,1] >= (self.voxelDims+self.buffer)/2, 5] = self.fiberDiffusions[0] #Store Diffusivity of the fibers in the 6-th dimension of the vector
            fiberCordinates[fiberCordinates[:,1] < (self.voxelDims+self.buffer)/2, 5] = self.fiberDiffusions[1]
            fiberCordinates_pre_rotation = fiberCordinates[fiberCordinates[:,1] <(self.voxelDims+self.buffer)/2, 0:3]
            rotatedCords = (self.rotMat.dot(fiberCordinates_pre_rotation.T)).T
            if rotatedCords.shape[0] > 0:
                z_correct = np.amin(rotatedCords[:,2]) # Want the grid to be placed at z = 0
                rotatedFibers = rotatedCords 
                rotatedFibers[:,2] = rotatedFibers[:,2] + np.abs(z_correct )
                fiberCordinates[fiberCordinates[:,1] < (self.voxelDims+self.buffer)/2, 0:3] = rotatedFibers   
            outputCords.append(fiberCordinates)
        outArg =  np.vstack([outputCords[0], outputCords[1]])
        return outArg
    
    def place_cell_grid(self):
        print('PLACING CELLS')
        numCells = self.numCells
        cellCentersTotal = []
        if not self.penetrating:
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
            Each Spin Must Know what Fiber it is in before distributing computation to the GPU. Thus, we pass an array containing the fiber index of spin the i-th spin to diffusion_in_fiber via the (numSpinInFiber, ) array
            fiberAtSpin_i.
        """
        if simulateFibers:
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
        
        if simulateCells:
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
        if simulateExtraEnvironment:
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
        
        """
        Global Variables

        spinInFiber_i - (numSpins,) array to be filled with the index of the fiber that the i-th spin is within (0 o.w.)
        spinInCell_i - (numSpins,) array to be filled with the index of the cell that the i-th spin is within (0 o.w.)
        initialSpinPositions - (numSpins,3) array with initial spin positions [READ ONLY]
        fiberCenters - (numFibers**2, 4) array with fiber locations and radii [READ ONLY]
        cellCenters - (numCells**3, 4) array with fiber locations and radii [READ ONLY]

        Local Variables

        Key - int() the location of the spin
        spinPosition - (3,) float(32), the current position of the spin

        RMK - spins in both a cell and a fiber are within the fiber. 

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

        """ 
        Global Variables:
        
        fiberCenters - a N x 4 matrix whose i-th row is a 4 vector containing the x,y,z position, and radius, of the fiber [READ ONLY]
        spinTrajectories - a N_spins x 3 matrix whose i = cuda.grid(1) element is handled by the thread launched from this kernel [READ AND WRITE]
        dt - time-step parameter (ms)
        Local Variables:

        D - the intrinsic diffusivity of the water in the cell
        prevPosition - a 3 vector (float32) containing the x,y,z cordinates of the previous position
        newPosition - a 3 vector (float32) containing the x,y,z cordinates of the proposed new position
        Step - Step size = Sqrt(6*D*dt)
        distanceFiber - (float32) distance between proposed new position and fibers intersecting the cell
        """
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
        
        """ 
        Global Variables:
        
        cellCenter - a 4 vector whose first 3 dimensions are the x,y,z cordinate of the cell, and whose last dimension is the cell radius [READ ONLY]
        fiberCenters - a  N x 4 matrix whose i-th row is a 4 vector containing the x,y,z position, and radius, of the fiber [READ ONLY]
        spinTrajectories - a N_spins x 3 matrix whose i = cuda.grid(1) element is handled by the thread launched from this kernel [READ AND WRITE]
        cellAtSpin_i - A (N_spins, ) array containing the index of the cell the spin is in 
        dt - time-step parameter (ms)
        
        Local Variables:

        D - the intrinsic diffusivity of the water in the cell
        prevPosition - a 3 vector (float32) containing the x,y,z cordinates of the previous position
        newPosition - a 3 vector (float32) containing the x,y,z cordinates of the proposed new position
        Step - Step size = Sqrt(6*D*dt)
        distanceCell - (float32) distance between proposed new position and radius of the cell
        distanceFiber - (float32) distance between proposed new position and fibers intersecting the cell
        inx2 - the index of the j-th penetrating fiber
        
        """
        inx = int32(cellAtSpin_i[i])
        D = float32(2.0)
        Step = float32(math.sqrt(6*D*dt))
        prevPosition = cuda.local.array(shape = 3, dtype= float32)
        newPosition = cuda.local.array(shape = 3, dtype= float32)
        distanceCell = float32(0.0)
        distanceFiber = float32(0.0)

        for step in range(numSteps):
            print(step)
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
        """ 
        Global Variables:
        
        cellCenters - a 4 vector whose first 3 dimensions are the x,y,z cordinate of the cell, and whose last dimension is the cell radius [READ ONLY]
        fiberCenters - a N x 4 matrix whose i-th row is a 4 vector containing the x,y,z position, and radius, of the fiber [READ ONLY]
        spinTrajectories - a N_spins x 3 matrix whose i = cuda.grid(1) element is handled by the thread launched from this kernel [READ AND WRITE]
        
        Local Variables:

        D - the intrinsic diffusivity of the water in the cell
        prevPosition - a 3 vector (float32) containing the x,y,z cordinates of the previous position
        newPosition - a 3 vector (float32) containing the x,y,z cordinates of the proposed new position
        Step - Step size = Sqrt(6*D*dt)
        distanceCell - (float32) distance between proposed new position and radius of the cell
        distanceFiber - (float32) distance between proposed new position and fibers intersecting the cell
        """
        D = float32(3.0)
        Step = float32(math.sqrt(6*D*dt))
        prevPosition = cuda.local.array(shape = 3, dtype= float32)
        newPosition = cuda.local.array(shape = 3, dtype= float32)
        distanceCell = float32(0.0)
        distanceFiber = float32(0.0)
       
        for step in range(numSteps): 
            print(step)
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
        traj1_vox = []
        traj2_vox = []

        for i in range(trajectoryT1m.shape[0]):
            if np.amin(trajectoryT2p[i,0:2]) >= 0 + 0.5*self.buffer and np.amax(trajectoryT2p[i,0:2]) <= self.voxelDims + 0.5*self.buffer:
                traj1_vox.append(trajectoryT1m[i,:])
                traj2_vox.append(trajectoryT2p[i,:])
        return np.array(traj1_vox), np.array(traj2_vox) 

    def signal(self, trajectoryT1m, trajectoryT2p, xyz):
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

    def save_data(self, path):
        
        overallData = []
        if self.simulateFibers:
            fiber_trajectories = [self.fiberPositionsT1m, self.fiberPositionsT2p]
            overallData.append(fiber_trajectories)
            print(self.fiberPositionsT1m.shape[0])
            np.save(path + os.sep + "fiberPositionsT1m_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.fiberPositionsT1m)
            np.save(path + os.sep + "fiberPositionsT2p_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.fiberPositionsT2p)
            pureFiberSignal, _ = self.signal(self.fiberPositionsT1m, self.fiberPositionsT2p, xyz = False)
            dwiFiber = nb.Nifti1Image(pureFiberSignal.reshape(1,1,1,-1), affine = np.eye(4))
            nb.save(dwiFiber, path + os.sep + "pureFiberSignal_angle={}_diffusivities={}_dt={}_ff={}.nii".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)))


        if self.simulateCells:
            cell_trajectories = [self.cellPositionsT1m, self.cellPositionsT2p]
            overallData.append(cell_trajectories)
            print(self.cellPositionsT1m.shape[0])
            np.save(path + os.sep + "cellPositionsT1m_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.cellPositionsT1m)
            np.save(path + os.sep + "cellPositionsT2p_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.cellPositionsT2p)
            pureCellSignal, _ = self.signal(self.cellPositionsT1m, self.cellPositionsT2p, xyz = False)
            dwiCell = nb.Nifti1Image(pureCellSignal.reshape(1,1,1,-1), affine = np.eye(4))
            nb.save(dwiCell, path + os.sep + "pureCellSignal_angle={}_diffusivities={}_dt={}_ff={}.nii".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)))
        if self.simulateExtra:
            water_trajectories = [self.extraPositionT1m, self.extraPositionT2p]
            overallData.append(water_trajectories)
            print(self.extraPositionT1m.shape[0])
            np.save(path + os.sep + "waterPositionsT1m_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.extraPositionT1m)
            np.save(path + os.sep + "waterPositionsT2p_angle={}_diffusivities={}_dt={}_ff={}.npy".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)), self.extraPositionT2p)
            pureWaterSignal, _ = self.signal(self.extraPositionT1m, self.extraPositionT2p, xyz = False)
            dwiWater = nb.Nifti1Image(pureWaterSignal.reshape(1,1,1,-1), affine = np.eye(4))
            nb.save(dwiWater, path + os.sep  + "pureWaterSignal_angle={}_diffusivities={}_dt={}_ff={}.nii".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)))
        
        if self.simulateFibers and self.simulateExtra:
            expSignal, bvals = self.signal(np.vstack([self.fiberPositionsT1m, self.extraPositionT1m]), np.vstack([self.fiberPositionsT2p, self.extraPositionT2p]), xyz = False)
            dwi = nb.Nifti1Image(expSignal.reshape(1,1,1,-1), affine = np.eye(4))
            nb.save(dwi,path + os.sep + "totalSignal_angle={}_diffusivities={}_dt={}_ff={}.nii".format(str(self.Thetas), str(self.fiberDiffusions), str(self.dt), str(self.fiberFraction)))

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
            axFiber = fig.add_subplot(projection = '3d')
            axFiber.scatter(self.fiberPositionsT2p[:,0], self.fiberPositionsT2p[:,1], self.fiberPositionsT2p[:,2], s = 1)
            axFiber.set_xlim(0,self.voxelDims+self.buffer)
            axFiber.set_ylim(0,self.voxelDims+self.buffer)
            axFiber.set_zlim(0,self.voxelDims+self.buffer)
            axFiber.set_xlabel('x')
            axFiber.set_ylabel('y')
            axFiber.set_zlabel('z')
            axFiber.legend()
        
        if plotExtra and self.simulateExtra:
            axExtra = fig.add_subplot(projection = '3d')
            axExtra.scatter(self.extraPositionT2p[:,0], self.extraPositionT2p[:,1], self.extraPositionT2p[:,2], s = 1)
            axExtra.set_xlim(0,self.voxelDims+self.buffer)
            axExtra.set_ylim(0,self.voxelDims+self.buffer)
            axExtra.set_zlim(0,self.voxelDims+self.buffer)
            axExtra.set_xlabel('x')
            axExtra.set_ylabel('y')
            axExtra.set_zlabel('z')
            axExtra.legend()

        if any([plotFibers, plotExtra, plotCells]):
            plt.show()


def main():   
    sim = dmri_simulation()
    Start = time.time()
    # inter-fiber-distance = 0, .1, .2, .5, 1.0
    # dt = .1, .001
    sim.set_parameters(
        numSpins= 1000*10**3,
        fiberFraction= (0.82, .82),  # Fraction in each Half/Quadrant Depending on 'P'/'NP'
        fiberRadius= 1.0,            # um
        Thetas = (0,0),              # degrees
        fiberDiffusions= (1.0, 1.0), # um^2/mm
        cellFraction= .0,            # Fraction in each Half/Quadrant Depending on 'P'/'NP'
        cellRadii= (3,10),           # um
        penetrating = 'P',           # 'P' = Penetrating Cells; 'NP = Non-Penetrating Cells 
        Delta = 10,                  # ms  
        dt = 0.0003,                    # ms 
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

        sim.simulate(simulateFibers=True, simulateCells=False, simulateExtraEnvironment=True)
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

        sWater, bWater = sim.signal(sim.extraPositionT1m, sim.extraPositionT2p, xyz = True)
        
        bwx, mwx = np.linalg.lstsq(A, np.log(sWater[0:20]), rcond=None)[0]
        bwy, mwy = np.linalg.lstsq(A, np.log(sWater[20:40]), rcond=None)[0]
        bwz, mwz = np.linalg.lstsq(A, np.log(sWater[40:60]), rcond=None)[0]
        plt.plot(bWater, (sWater[0:20]), 'bx', label = mwx)
        plt.plot(bWater, (sWater[20:40]), 'rx', label = mwy)
        plt.plot(bWater, (sWater[40:60]), 'gx', label = mwz)
        plt.legend()
        plt.savefig(r"C:\MCSIM\dMRI-MCSIM-main\gpu_data\extra_signal_angle={}_diffusivities={}_dt={}_ff={}_dist={}.png".format(sim.Thetas, sim.fiberDiffusions,sim.dt,sim.fiberFraction, round(ifd,4)))

        plt.clf()
        End = time.time()
        sim.plot(plotFibers=False, plotCells=False,plotExtra=True)
        print('Simulation Executed in: {} sec'.format(End-Start))
    return
   



if __name__ == "__main__":
    main()

    
 



