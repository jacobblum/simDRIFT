from re import A
from turtle import pen
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
        Delta = 0.0 #ms
        dt = 0.0 #ms
        delta = 0.0 #ms
        numCells = 0.0
        cellFraction = 0.0
        cellRadius = 0.0 #um
        spinPositionsT1m = 0
        fiberPositionsT1m = 0
        fiberPositionsT2p = 0
        cellPositionsT1m = 0
        cellPositionsT2p = 0
        extraPositionsT1m = 0
        extraPositionsT2p = 0
        spinInFiber_i = 0
        spinInCell_i = 0
        return
    
    def set_parameters(self, numSpins, fiberFraction, fiberRadius, cellFraction, cellRadius, Delta, dt, voxelDim):
        self.voxelDims = voxelDim
        self.numSpins = numSpins
        self.fiberFraction = fiberFraction
        self.fiberRadius = fiberRadius
        self.numFibers = self.set_num_fibers()
        self.cellFraction = cellFraction
        self.cellRadius = cellRadius
        self.numCells = self.set_num_cells()
        self.Delta = Delta
        self.dt = dt
        self.delta = dt
        self.fiberCenters = self.place_fiber_grid()
        self.cellCenters = self.place_cell_grid()
        self.spinPotionsT1m = np.random.uniform(low = 0, high = 200, size = (int(self.numSpins),3))
        self.spinInFiber_i = -1*np.ones(self.numSpins)
        self.spinInCell_i = -1*np.ones(self.numSpins)
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
        spinInFiber_i_GPU = cuda.to_device(self.spinInFiber_i.astype(np.float32))
        spinInCell_i_GPU = cuda.to_device(self.spinInCell_i.astype(np.float32))
        spinInitialPositions_GPU = cuda.to_device(self.spinPotionsT1m.astype(np.float32))
        fiberCenters_GPU = cuda.to_device(self.fiberCenters.astype(np.float32))
        cellCenters_GPU = cuda.to_device(self.cellCenters.astype(np.float32))
        Start = time.time()
        self.find_spin_locations.forall(self.numSpins)(spinInFiber_i_GPU, spinInCell_i_GPU, spinInitialPositions_GPU, fiberCenters_GPU, cellCenters_GPU)
        End = time.time()
        print('Finding {} spins - task completed in {} sec'.format(self.numSpins, End-Start))
        self.spinInFiber_i = spinInFiber_i_GPU.copy_to_host()
        self.spinInCell_i = spinInCell_i_GPU.copy_to_host()

       


    def simulate(self, simulateFibers, simulateCells, simulateWater):
        """
        Simulate Fiber Diffusion: 
            Each Spin Must Know what Fiber it is in before distributing computation to the GPU. Thus, we pass an array containing the fiber index of spin the i-th spin to diffusion_in_fiber via the (numSpinInFiber, ) array
            fiberAtSpin_i.
        """
        if simulateFibers:
            fiberSpins, self.fiberPositionsT1m = self.spinPotionsT1m[self.spinInFiber_i != -1].astype(np.float32), self.spinPotionsT1m[self.spinInFiber_i != -1].astype(np.float32)
            fiberAtSpin_i = self.spinInFiber_i[self.spinInFiber_i != -1].astype(np.float32)
            rng_states_fibers = create_xoroshiro128p_states(len(self.spinInFiber_i[self.spinInFiber_i != -1]), seed = 42)
            fiberCenters_GPU = cuda.to_device(self.fiberCenters.astype(np.float32))
            Start = time.time()
            self.diffusion_in_fiber.forall(len(fiberAtSpin_i))(rng_states_fibers,
                                    fiberSpins,
                                    fiberAtSpin_i,
                                    int(self.Delta/self.dt),
                                    fiberCenters_GPU,
                                    self.dt)
            End = time.time()
            self.fiberPositionsT2p = fiberSpins
            print('Fiber Diffusion Compuation Time: {} seconds'.format(End-Start))
            
        
        """
        Simulate Intra-Cellular Diffusion: 
            Each Spin must know which cell it is in before distributing computation to the GPU. Also, to avoid looping over all of the fibers, we need to also pass an array of penetrating fiber indicies to diffusion_in_cells
        """
        
        if simulateCells:
            cellSpins = self.spinPotionsT1m[(self.spinInCell_i > -1) & (self.spinInFiber_i < 0)] 
            self.cellPositionsT1m = cellSpins.copy()
            cellSpins_GPU = cuda.to_device(cellSpins)
            penetratingFibers_GPU = cuda.to_device(np.zeros((self.numCells**2,int(self.cellRadius**2/self.fiberRadius**2))))
            cellAtSpin_i_GPU = cuda.to_device(self.spinInCell_i[(self.spinInCell_i > -1) & (self.spinInFiber_i < 0)])
            self.find_penetrating_fibers.forall(self.numCells**2)(penetratingFibers_GPU, self.cellCenters, self.fiberCenters)
            
            rng_states_cells = create_xoroshiro128p_states(cellSpins.shape[0], seed = 42)
            Start = time.time()
            self.diffusion_in_cell.forall(cellSpins.shape[0])(rng_states_fibers, 
                                    cellSpins_GPU,
                                    cellAtSpin_i_GPU,
                                    int(self.Delta/self.dt),
                                    self.cellCenters,
                                    penetratingFibers_GPU,
                                    fiberCenters_GPU,
                                    self.dt
            )
            End = time.time()
            self.cellPositionsT2p = cellSpins_GPU.copy_to_host()
            print('Cell Diffusion Computation Time: {} seconds'.format(End - Start))
        """
        Simulate Extra-Cellular and Extra-Axonal Diffusion
        """
        if simulateWater:
            self.extraPositionT1m =(self.spinPotionsT1m[(self.spinInCell_i < 0) & (self.spinInFiber_i < 0)])
            rng_states_Extra = create_xoroshiro128p_states(self.extraPositionT1m.shape[0], seed = 42)
            extraSpins_GPU = cuda.to_device(self.extraPositionT1m.astype(np.float32))
            
            Start = time.time()
            self.diffusion_in_water.forall(self.extraPositionT1m.shape[0])(rng_states_Extra,
                                            extraSpins_GPU,
                                            int(self.Delta/self.dt),
                                            self.cellCenters,
                                            fiberCenters_GPU,
                                            self.dt
            )
            End = time.time()
            self.extraPositionT2p = extraSpins_GPU.copy_to_host()
            print('Extra Cellular Diffusion Computation Time: {} seconds'.format(End - Start))


    @cuda.jit
    def find_penetrating_fibers(penetratingFibers, cellCenters, fiberCenters):
        i = cuda.grid(1)
        if i > penetratingFibers.shape[0]:
            return
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
        distanceFiber = float32(0.0)
        indexList = cuda.local.array(shape = 3, dtype= float32)
        for k in range(indexList.shape[0]): indexList[k] = -1
        ctr = int32(0)
        for j in range(fiberCenters.shape[0]):
            distanceFiber = jp.euclidean_distance(cellCenters[i,0:3], fiberCenters[j,0:3], 'xy')
            if distanceFiber < cellCenters[i,3]:
                indexList[ctr] = j
                ctr += 1
        cuda.syncthreads()
        for k in range(penetratingFibers.shape[1]): penetratingFibers[i,k] = indexList[k]
        cuda.syncthreads()
        return       

    @cuda.jit 
    def find_spin_locations(spinInFiber_i, spinInCell_i, initialSpinPositions, fiberCenters, cellCenters):
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
        Key = int32(-1)
        spinPosition = cuda.local.array(shape = 3, dtype = float32)
        fiberDistance = float32(0.0)
        cellDistance = float32(0.0)

        for k in range(spinPosition.shape[0]): spinPosition[k] = initialSpinPositions[i,k]
        for j in range(fiberCenters.shape[0]): 
            fiberDistance = jp.euclidean_distance(spinPosition, fiberCenters[j,0:3], 'xy')
            if fiberDistance < fiberCenters[j,3]: 
                KeyFiber = j
                break
        
        for j in range(cellCenters.shape[0]):
            cellDistance = jp.euclidean_distance(spinPosition, cellCenters[j,0:3], 'xyz')
            if cellDistance < cellCenters[j,3]:
                KeyCell = j
                break
        
        cuda.syncthreads()
        spinInCell_i[i] = KeyCell
        spinInFiber_i[i] = KeyFiber
        cuda.syncthreads()
        return
            
    @cuda.jit
    def diffusion_in_fiber(rng_states, spinTrajectories, fiberIndexAt_i, numSteps, fiberCenters, dt):
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
        D = float32(1.0)
        Step = float32(math.sqrt(6*D*dt))
        prevPosition = cuda.local.array(shape = 3, dtype= float32)
        newPosition = cuda.local.array(shape = 3, dtype = float32)
        distanceFiber = float32(0.0)
    
        for step in range(numSteps):
            newPosition = jp.randomDirection(rng_states, newPosition, i)
            for k in range(newPosition.shape[0]): 
                prevPosition[k] = spinTrajectories[i,k] 
                newPosition[k] = prevPosition[k] + (Step * newPosition[k])
            distanceFiber = jp.euclidean_distance(newPosition,fiberCenters[inx,0:3], 'xy')
            if distanceFiber > fiberCenters[inx,3]:
                for k in range(newPosition.shape[0]): newPosition[k] = prevPosition[k]
            cuda.syncthreads()
            for k in range(newPosition.shape[0]): spinTrajectories[i,k] = newPosition[k]
            cuda.syncthreads()
        return

    @cuda.jit
    def diffusion_in_cell(rng_states, spinTrajecotires, cellAtSpin_i, numSteps, cellCenters, penetratingFibers, fiberCenters, dt):
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
            distanceCell = jp.euclidean_distance(newPosition, cellCenters[inx,0:3], 'xyz')
            if distanceCell > cellCenters[inx,3]:
                for k in range(newPosition.shape[0]): newPosition[k] = prevPosition[k]
            else:
                for j in range(penetratingFibers[i,:].shape[0]):
                    inx2 = int(penetratingFibers[i,j])
                    if inx2 > -1:
                        distanceFiber = jp.euclidean_distance(newPosition, fiberCenters[inx2,0:3], 'xy')
                        if distanceFiber < fiberCenters[inx2,3]:
                            for k in range(newPosition.shape[0]): newPosition[k] = prevPosition[k]
                            break
            
            cuda.syncthreads()
            for k in range(newPosition.shape[0]): spinTrajecotires[i,k] = newPosition[k]
            cuda.syncthreads() 

            if ((i == 0) & (step%200 == 0)):
                print('Diffusion in Cell: Block 1 Thread 0 Progress: ', step, '/', numSteps-1)   
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
                print('Diffusion in Water: Block 1 Thread 0 Progress: ', step, '/', numSteps-1)       
        return 

    def signal(self, trajectoryT1m, trajectoryT2p):
        gamma = 42.58
        Delta = self.Delta #ms
        dt = self.delta # ms 
        delta = dt #ms
        b_vals = np.linspace(0, 1500, 20)
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
        return allSignal, b_vals


def main():
    
    sim = dmri_simulation()
    Start = time.time()
    sim.set_parameters(
        numSpins= 100*10**3,
        fiberFraction= .05,
        fiberRadius= 1.0,
        cellFraction= .10,
        cellRadius= 5.0,
        Delta = 20,
        dt = .001,
        voxelDim=200
        )
    sim.simulate(simulateFibers=True, simulateCells=True, simulateWater=False)
    End = time.time()
    print('Simulation Executed in {} seconds'.format(End-Start))
    fiber_signal, bvals = sim.signal(sim.fiberPositionsT1m, sim.fiberPositionsT2p)
    fig, ax = plt.subplots(figsize = (10,3))
    A = np.vstack([np.ones(len(bvals)), -1*bvals]).T
    _, D = np.linalg.lstsq(A, np.log(fiber_signal[40:60]), rcond=None)[0]
    print(D * 10**3)

    ax.plot(bvals, np.log(fiber_signal[0:20]), 'rx', label = 'x')
    ax.plot(bvals, np.log(fiber_signal[20:40]), 'bx', label = 'y')
    ax.plot(bvals, np.log(fiber_signal[40:60]), 'gx', label = 'z')
    
    ax.set_xlabel(r'$b$')
    ax.set_ylabel(r'$s_{k}$')
    ax.set_title(r'Estimated Diffusion: {}'.format(round(D*10**3,4)))
    ax.legend()
    plt.show()    

if __name__ == "__main__":
    main()

    
 



