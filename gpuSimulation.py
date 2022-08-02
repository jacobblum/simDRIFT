import numpy as np 
import numba 
from numba import jit, cuda, int32, void, float32
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import math
from MCSIMplots import plot
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
        fiberRotationReference = 0
        Thetas = 0
        rotMat = 0
        return
    
    def set_parameters(self, numSpins, fiberFraction, fiberRadius, Thetas, cellFraction, cellRadius, Delta, dt, voxelDim):
        self.voxelDims = voxelDim
        self.numSpins = numSpins
        self.fiberFraction = fiberFraction
        self.fiberRadius = fiberRadius
        self.Thetas = Thetas
        self.fiberRotationReference = self.rotation()        
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
        fiberCordinates = np.zeros((self.numFibers**2,5))
        fibers1d = np.linspace(0+self.fiberRadius, self.voxelDims-self.fiberRadius, self.numFibers)
        fibers2d = np.outer(fibers1d, np.ones(fibers1d.shape[0]))
        fiberXs, fiberYs = fibers2d.T, fibers2d
        fiberCordinates[:,0] = fiberXs.flatten()
        fiberCordinates[:,1] = fiberYs.flatten()
        fiberCordinates[:,2] = 1.0
        fiberCordinates[:,3] = self.fiberRadius
        fiberCordinates[fiberCordinates[:,1] < 100, 4] = 1
        fiberCordinates_pre_rotation = fiberCordinates[fiberCordinates[:,1] < 100, 0:3]

        rotatedCords = (self.rotMat.dot(fiberCordinates_pre_rotation.T)).T
        z_correct = np.amin(rotatedCords[:,2])
        rotatedFibers = rotatedCords 
        rotatedFibers[:,2] = rotatedFibers[:,2] + np.abs( z_correct )
        fiberCordinates[fiberCordinates[:,1] < 100, 0:3] = rotatedFibers 
        
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
        spinInFiber_i_GPU = (self.spinInFiber_i.astype(np.float32))
        spinInCell_i_GPU  = (self.spinInCell_i.astype(np.float32))
        spinInitialPositions_GPU = (self.spinPotionsT1m.astype(np.float32))
        fiberCenters_GPU = (self.fiberCenters.astype(np.float32))
        cellCenters_GPU = (self.cellCenters.astype(np.float32))
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
        print(rotationReferences)

        return rotationReferences

    
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
                                    self.fiberRotationReference,
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
            cellSpins_GPU = (cellSpins)
            penetratingFibers_GPU = (np.zeros((self.numCells**2,int(self.cellRadius**2/self.fiberRadius**2))))
            cellAtSpin_i_GPU = (self.spinInCell_i[(self.spinInCell_i > -1) & (self.spinInFiber_i < 0)])
            
           # test = self.find_penetrating_fibers(self.cellCenters, self.fiberCenters, self.fiberRotationReference) ## Surprisingly the computation wasn't hurt by just iterating through all the fibers?
            
            rng_states_cells = create_xoroshiro128p_states(cellSpins.shape[0], seed = 42)
            Start = time.time()
            self.diffusion_in_cell.forall(cellSpins.shape[0])(rng_states_cells, 
                                    cellSpins_GPU,
                                    cellAtSpin_i_GPU,
                                    int(self.Delta/self.dt),
                                    self.cellCenters,
                                    penetratingFibers_GPU,
                                    fiberCenters_GPU,
                                    self.fiberRotationReference,
                                    self.dt
            )
            End = time.time()
            self.cellPositionsT2p = cellSpins_GPU
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
                                            self.fiberRotationReference,
                                            self.dt
            )
            End = time.time()
            self.extraPositionT2p = extraSpins_GPU.copy_to_host()
            print('Extra Cellular Diffusion Computation Time: {} seconds'.format(End - Start))


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
        Key = int32(-1)
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
        D = float32(1.0)
        Step = float32(math.sqrt(6*D*dt))
        prevPosition = cuda.local.array(shape = 3, dtype= float32)
        newPosition = cuda.local.array(shape = 3, dtype = float32)
        distanceFiber = float32(0.0)
        rotationIndex = int(fiberCenters[inx, 4])

        for step in range(numSteps):
            newPosition = jp.randomDirection(rng_states, newPosition, i)
            for k in range(newPosition.shape[0]): 
                prevPosition[k] = spinTrajectories[i,k] 
                newPosition[k] = prevPosition[k] + (Step * newPosition[k])
            distanceFiber = jp.euclidean_distance(newPosition,fiberCenters[inx,0:3], fiberRotationReference[rotationIndex,:], 'fiber')
            if distanceFiber > fiberCenters[inx,3]:
                for k in range(newPosition.shape[0]): newPosition[k] = prevPosition[k]
            cuda.syncthreads()
            for k in range(newPosition.shape[0]): spinTrajectories[i,k] = newPosition[k]
            cuda.syncthreads()
        return

    @cuda.jit
    def diffusion_in_cell(rng_states, spinTrajecotires, cellAtSpin_i, numSteps, cellCenters, penetratingFibers, fiberCenters, fiberRotationReference, dt):
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
            distanceCell = jp.euclidean_distance(newPosition, cellCenters[inx,0:3], fiberRotationReference[0,:], 'cell')
            if distanceCell > cellCenters[inx,3]:
                for k in range(newPosition.shape[0]): newPosition[k] = prevPosition[k]
            else:
                for j in range(fiberCenters.shape[0]):
                    rotation_Index = int(fiberCenters[j,4])
                    distanceFiber = jp.euclidean_distance(newPosition, fiberCenters[j,0:3], fiberRotationReference[rotation_Index,:], 'fiber')
                    if distanceFiber < fiberCenters[j,3]:
                        for k in range(newPosition.shape[0]): newPosition[k] = prevPosition[k]
                        break
            
            cuda.syncthreads()
            for k in range(newPosition.shape[0]): spinTrajecotires[i,k] = newPosition[k]
            cuda.syncthreads() 

            if ((i == 0)):
                print('Diffusion in Cell: Block 1 Thread 0 Progress: ', step, '/', numSteps-1)   
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
            newPosition = jp.randomDirection(rng_states, newPosition, i)        
            for k in range(newPosition.shape[0]): 
                prevPosition[k] = spinTrajectories[i,k]
                newPosition[k] = prevPosition[k] + Step * newPosition[k]
            
            for l in range(cellCenters.shape[0]):
                distanceCell = jp.euclidean_distance(newPosition, cellCenters[l, 0:3], fiberRotationReference, 'cell')
                if distanceCell < cellCenters[l,3]:
                    for k in range(newPosition.shape[0]): newPosition[k] = prevPosition[k]
            for l in range(fiberCenters.shape[0]):
                distanceFiber = jp.euclidean_distance(newPosition, fiberCenters[l,0:3], fiberRotationReference, 'fiber')
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
    
    def plot(self):
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
        ax = fig.add_subplot(projection = '3d')
        #for i in range(self.fiberCenters.shape[0]):
        #    color = ['red', 'blue']
        #    x,y,z = plot_fiber(self, self.fiberCenters[i,:])
        #    ax.plot_surface(x,y,z, color = color[int(self.fiberCenters[i,4])])
        ax.scatter(self.fiberCenters[:,0], self.fiberCenters[:,1], self.fiberCenters[:,2])


        ax.scatter(self.fiberPositionsT2p[self.fiberPositionsT1m[:,1] < 95,0], self.fiberPositionsT2p[self.fiberPositionsT1m[:,1] < 95,1], self.fiberPositionsT2p[self.fiberPositionsT1m[:,1] < 95,2], color = 'red')
        ax.scatter(self.fiberPositionsT2p[self.fiberPositionsT1m[:,1] >= 95,0], self.fiberPositionsT2p[self.fiberPositionsT1m[:,1] >= 95,1], self.fiberPositionsT2p[self.fiberPositionsT1m[:,1] >= 95,2], color = 'blue')
        #ax.scatter(self.cellPositionsT2p[:,0], self.cellPositionsT2p[:,1], self.cellPositionsT2p[:,2])
        ax.set_xlim(0,200)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


def main():
    
    sim = dmri_simulation()
    Start = time.time()
    sim.set_parameters(
        numSpins= 100*10**3,
        fiberFraction= .10,
        fiberRadius= 5.0,
        Thetas = (0,90),
        cellFraction= .20,
        cellRadius= 20.0,
        Delta = 20,
        dt = .005,
        voxelDim=200
        )
    
    sim.simulate(simulateFibers=True, simulateCells=False, simulateWater=False)
    End = time.time()
    print('Simulation Executed in {} seconds'.format(End-Start))
    sim.plot()

    exit()


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

    
 



