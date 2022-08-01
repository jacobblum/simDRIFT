from concurrent.futures import thread
from re import L
from turtle import distance
from matplotlib import projections
import numpy as np 
import numba 
from numba import jit, cuda, int32, void, float32
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math
import jp
import matplotlib.pyplot as plt
import time 
from tqdm import tqdm





class dmri_simulation:
    def __init__(self):
        numSpins = 0.0 
        numFibers = 0.0
        fiberFraction = 0.0
        fiberRadius = 0.0 #um
        TE = 0.0 #ms
        dt = 0.0 #ms
        delta = 0.0 #ms
        numCells = 0.0
        cellFraction = 0.0
        cellRadius = 0.0 #um
        spinPositionsT1m = 0
        spinLocation = 0

        return
    
    def set_parameters(self, numSpins, fiberFraction, fiberRadius, cellFraction, cellRadius, TE, dt, voxelDim):
        self.voxelDims = voxelDim
        self.numSpins = numSpins
        self.fiberFraction = fiberFraction
        self.fiberRadius = fiberRadius
        self.numFibers = self.set_num_fibers()
        self.cellFraction = cellFraction
        self.cellRadius = cellRadius
        self.numCells = self.set_num_cells()
        self.TE = TE
        self.dt = dt
        self.delta = dt
        self.fiberCenters = self.place_fiber_grid()
        self.cellCenters = self.place_cell_grid()
        self.spinPotionsT1m = np.random.uniform(low = 0, high = 200, size = (int(self.numSpins),3))
        self.spinLocation = np.zeros(self.numSpins)
        self.get_spin_locations()



    def set_num_fibers(self):
        fiberFraction = self.fiberFraction
        fiberRadius = self.fiberRadius
        numFiber = int(np.sqrt((fiberFraction * self.voxelDims**2)/(np.pi*fiberRadius**2)))
        return numFiber
    
    def set_num_cells(self):
        cellFraction = self.cellFraction
        cellRadius = self.cellRadius
        numCells = int(np.cbrt((cellFraction*self.voxelDims**3)/((4/3)*np.pi*cellRadius**3)))
        return numCells
    
    def place_fiber_grid(self):
        fiberCordinates = np.zeros((self.numFibers**2,4))
        fibers1d = np.linspace(0+self.fiberRadius, self.voxelDims-self.fiberRadius, self.numFibers)
        fibers2d = np.outer(fibers1d, np.ones(fibers1d.shape[0]))
        fiberXs, fiberYs = fibers2d.T, fibers2d
        fiberCordinates[:,0] = fiberXs.flatten()
        fiberCordinates[:,1] = fiberYs.flatten()
        fiberCordinates[:,3] = self.fiberRadius
        return fiberCordinates
    
    def place_cell_grid(self):
        numCells = self.numCells
        cellCenters = np.zeros((numCells**3, 4))
        cellXs, cellYs, cellZs = np.mgrid[0 + self.cellRadius: self.voxelDims-self.cellRadius:(numCells *1j),0 + self.cellRadius: self.voxelDims-self.cellRadius:(numCells *1j), 0 + self.cellRadius: self.voxelDims-self.cellRadius:(numCells *1j)]
        cellCenters[:,0] = cellXs.flatten()
        cellCenters[:,1] = cellYs.flatten()
        cellCenters[:,2] = cellZs.flatten()
        cellCenters[:,3] = self.cellRadius
        return cellCenters
    
    def get_spin_locations(self):
        spinLocations_GPU = cuda.to_device(self.spinLocation.astype(np.float32))
        spinInitialPositions_GPU = cuda.to_device(self.spinPotionsT1m.astype(np.float32))
        fiberCenters_GPU = cuda.to_device(self.fiberCenters.astype(np.float32))
        cellCenters_GPU = cuda.to_device(self.cellCenters.astype(np.float32))
        Start = time.time()
        self.find_spin_locations.forall(self.numSpins)(spinLocations_GPU, spinInitialPositions_GPU, fiberCenters_GPU, cellCenters_GPU)
        End = time.time()
        print('Finding {} spins - task completed in {} sec'.format(self.numSpins, End-Start))
        self.spinLocation = spinLocations_GPU.copy_to_host()

        print(self.spinLocation[self.spinLocation == 1].shape)

    @cuda.jit 
    def find_spin_locations(spinLocations, initialSpinPositions, fiberCenters, cellCenters):
        i = cuda.grid(1)
        if i > len(spinLocations):
            return 
        
        """
        Global Variables

        spinLocations - (numSpins,) array containing the spin location key: 0 = in free water, 1 = in fiber, 2 = in cell [Write]
        initialSpinPositions - (numSpins,3) array with initial spin positions [READ ONLY]
        fiberCenters - (numFibers**2, 4) array with fiber locations and radii [READ ONLY]
        cellCenters - (numCells**3, 4) array with fiber locations and radii [READ ONLY]

        Local Variables

        Key - int() the location of the spin
        spinPosition - (3,) float(32), the current position of the spin

        RMK - spins in both a cell and a fiber are within the fiber. 

        """
        KeyFiber = int32(0)
        KeyCell = int32(0)
        Key = int32(0)
        spinPosition = cuda.local.array(shape = 3, dtype = float32)
        fiberDistance = float32(0.0)
        cellDistance = float32(0.0)

        for k in range(spinPosition.shape[0]): spinPosition[k] = initialSpinPositions[i,k]
        for j in range(fiberCenters.shape[0]): 
            fiberDistance = jp.euclidean_distance(spinPosition, fiberCenters[j,0:3], 'xy')
            if fiberDistance < fiberCenters[j,3]: 
                KeyFiber = 1
                break
        
        for j in range(cellCenters.shape[0]):
            cellDistance = jp.euclidean_distance(spinPosition, cellCenters[j,0:3], 'xyz')
            if cellDistance < cellCenters[j,3]:
                KeyCell = 2
                break

        
        if (KeyFiber == 1) & (KeyCell == 2):
            Key = KeyFiber
        elif (KeyFiber == 1) & (KeyCell == 0):
            Key = KeyFiber
        elif (KeyFiber == 0) & (KeyCell == 2):
            Key = KeyCell
        else:
            Key = 0
        
        cuda.syncthreads()
        spinLocations[i] = Key
        cuda.syncthreads()
        return
            




    @cuda.jit
    def diffusion_in_fiber(rng_states, spinTrajectories, numSteps, fiberCenter, dt):
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
        
        D = float32(2.0)
        Step = float32(math.sqrt(6*D*dt))
        prevPosition = cuda.local.array(shape = 3, dtype= float32)
        newPosition = cuda.local.array(shape = 3, dtype = float32)
        distanceFiber = float32(0.0)
    
        for step in range(numSteps):
            newPosition = jp.randomDirection(rng_states, newPosition, i)
            for k in range(newPosition.shape[0]): 
                prevPosition[k] = spinTrajectories[i,k] 
                newPosition[k] = prevPosition[k] + (Step * newPosition[k])
            distanceFiber = jp.euclidean_distance(newPosition,fiberCenter[0:3], 'xy')
            if distanceFiber > fiberCenter[3]:
                for k in range(newPosition.shape[0]): newPosition[k] = prevPosition[k]
            cuda.syncthreads()
            for k in range(newPosition.shape[0]): spinTrajectories[i,k] = newPosition[k]
            cuda.syncthreads()
        return

    @cuda.jit
    def diffusion_in_cell(rng_states, spinTrajecotires, numSteps, cellCenter, fiberCenters, dt):
        i = cuda.grid(1)
        if i > spinTrajecotires.shape[0]:
            return
        
        """ 
        Global Variables:
        
        cellCenter - a 4 vector whose first 3 dimensions are the x,y,z cordinate of the cell, and whose last dimension is the cell radius [READ ONLY]
        fiberCenters - a  N x 4 matrix whose i-th row is a 4 vector containing the x,y,z position, and radius, of the fiber [READ ONLY]
        spinTrajectories - a N_spins x 3 matrix whose i = cuda.grid(1) element is handled by the thread launched from this kernel [READ AND WRITE]
        dt - time-step parameter (ms)
        
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
            newPosition = jp.randomDirection(rng_states, newPosition, i)
            for k in range(newPosition.shape[0]):
                prevPosition[k] = spinTrajecotires[i,k]
                newPosition[k] = prevPosition[k] + Step*newPosition[k]
            distanceCell = jp.euclidean_distance(newPosition, cellCenter[0:3], 'xyz')
            if distanceCell > cellCenter[3]:
                for k in range(newPosition.shape[0]): newPosition[k] = prevPosition[k]
            else:
                for j in range(fiberCenters.shape[0]):
                    distanceFiber = jp.euclidean_distance(newPosition, fiberCenters[j,0:3], 'xy')
                    if distanceFiber < fiberCenters[j,3]:
                        for k in range(newPosition.shape[0]): newPosition[k] = prevPosition[k]
                        break
            
            cuda.syncthreads()
            for k in range(newPosition.shape[0]): spinTrajecotires[i,k] = newPosition[k]
            cuda.syncthreads() 
        return 

    @cuda.jit 
    def diffusion_in_water(rng_states, spinTrajectories, numSteps, cellCenters, fiberCenters, dt):
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
            newPosition = jp.randomDirection(rng_states, newPosition, i)        
            for k in range(newPosition.shape[0]): 
                prevPosition[k] = spinTrajectories[i,k]
                newPosition[k] = prevPosition[k] + Step * newPosition[k]
            
            for l in range(cellCenters.shape[0]):
                distanceCell = jp.euclidean_distance(newPosition, cellCenters[l, 0:3], 'xyz')
                if distanceCell < cellCenters[l,3]:
                    for k in range(newPosition.shape[0]): newPosition[k] = prevPosition[k]
            for l in range(fiberCenters.shape[0]):
                distanceFiber = jp.euclidean_distance(newPosition, fiberCenters[l,0:3], 'xy')
                if distanceFiber < fiberCenters[l,3]:
                    for k in range(newPosition.shape[0]): newPosition[k] = prevPosition[k]
            
            cuda.syncthreads()
            for k in range(newPosition.shape[0]): spinTrajectories[i,k] = newPosition[k]
            cuda.syncthreads() 
            if ((i == 0) & (step%200 == 0)):
                print('Block 1 Thread 0 Progress: ', step, '/', numSteps-1)       
        return 

def signal(trajectoryT1m, trajectoryT2p):
    gamma = 42.58
    Delta = 20 #ms
    dt = .0050 # ms 
    delta = dt #ms
    b_vals = np.linspace(0, 3000, 20)
    Gt = np.sqrt(10**-3 * b_vals/(gamma**2 * delta**2*(Delta-delta/3)))
    unitGradients = np.zeros((3*len(Gt), 3))
    for i in (range(unitGradients.shape[1])):
        unitGradients[i*len(b_vals): (i+1)*len(b_vals),i] = Gt

    allSignal = np.zeros(unitGradients.shape[0])
    for i in tqdm(range(unitGradients.shape[0])):
        signal = 0
        for j in range(trajectoryT1m.shape[0]):
            phase_shift = gamma * np.sum(unitGradients[i,:].dot(trajectoryT1m[j,:]-trajectoryT2p[j,:])) * dt
            signal = signal + np.exp(-1 *(0+1j) * phase_shift)
        signal = signal/trajectoryT1m.shape[0]
        allSignal[i] = np.abs(signal)
    return allSignal[40:]



def main():
    
    sim = dmri_simulation()
    sim.set_parameters(
        numSpins=100*10**3,
        fiberFraction= .35,
        fiberRadius= 1.0,
        cellFraction= .10,
        cellRadius= 5.0,
        TE = 20,
        dt = .005,
        voxelDim=200
        )

    exit()

    
    num_spins = 100 * 10**3
    #spinTrajectories = (np.random.uniform(low = 0, high = 200, size = (num_spins, 3))).astype(np.float32)
    


    spinTrajectories = 11*np.ones((num_spins,3)).astype(np.float32)
    originalPositions = spinTrajectories.copy()
    spinTrajectories_GPU = cuda.to_device(spinTrajectories)
    Delta = 20 #ms
    dt = 0.005 #ms

    numSteps = int(Delta/dt)
    cellCenter = np.array([[0,0,0,0.0], [0,0,0,0.0]])
    fiberCenters = (np.ones((65**2, 4)).astype(np.float32))
    fiberCenters[:, 3] = 10

    fiberCenters = cuda.to_device(fiberCenters)


    eixt()

   
    
    
    rng_states = create_xoroshiro128p_states(num_spins * 1, seed = 42)
    start = time.time()
    sim.diffusion_in_water.forall(spinTrajectories.shape[0])(rng_states, spinTrajectories_GPU, numSteps, cellCenter, fiberCenters, dt)
    end = time.time()
        
    print('Simulation Time: {} sec'.format(end - start))
    

    spinTrajectories_end = spinTrajectories_GPU.copy_to_host()
    
    print('Simulation Time: {} sec'.format(end - start))



    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(spinTrajectories_end[:,0], spinTrajectories_end[:,1], spinTrajectories_end[:,2])
    plt.show()

    exit()

    testSignal = signal(originalPositions, spinTrajectories_end)
    bvals = np.linspace(0,1500,20)
    A = np.vstack([np.ones(len(bvals)), -1*bvals]).T
    
    _, D = np.linalg.lstsq(A, np.log(testSignal), rcond=None)[0]
    print(D * 10**3)
    
    fig, ax = plt.subplots(figsize = (7,7))

    ax.plot(np.linspace(0,1500, 20), (testSignal))
    ax.set_xlabel(r'$b$')
    ax.set_ylabel(r'$\log(s)$')
    ax.set_title('Estimated D: {}'.format(str(round(D,7)*10**3)))
    plt.show()

 



if __name__ == "__main__":
    main()

    
 



