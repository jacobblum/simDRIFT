from cProfile import label
import numpy as np 
import matplotlib.pyplot as plt
import time 
import os  
import nibabel as nb
import glob as glob 
import configparser
from ast import literal_eval
from multiprocessing import Process
import numba 
from numba import jit, cuda, int32, float32
from numba.cuda import random 
from numba.cuda.random import xoroshiro128p_normal_float32,  create_xoroshiro128p_states
import jp as jp

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
        buffer = 0
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
        #self.path_to_save = config['Saving Parameters']['path_to_save_file_dir']
        self.path_to_save = 'C:\MCSIM\ISRM'

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
        self.plot(plotFibers=False, plotCells=False, plotExtra=False, plotConfig=False)
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
    
        if self.fiberCofiguration == 'Interwoven' or self.fiberCofiguration == 'IW':
            Ys_mod2 = np.unique(outputArg[:,1])[::2]
            idx = (np.in1d(outputArg[:,1], Ys_mod2))
            fiberCordinates_pre_rotation = outputArg[idx, 0:3]

        if self.fiberCofiguration == 'Penetrating' or self.fiberCofiguration == 'Void' or self.fiberCofiguration == 'P':
            idx = np.where(outputArg[:,1] < 0.5*(self.voxelDims+self.buffer))[0]
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
        Start = time.time()
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
        End = time.time()
        print('Cell Population Time: {} seconds'.format(End-Start))
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
            plt.show()

        axFiber = fig.add_subplot(projection = '3d')

        if plotFibers:
            axFiber.scatter(self.fiberCenters[:,0], self.fiberCenters[:,1], self.fiberCenters[:,2])      
            axFiber.set_xlim(0.5*self.buffer,self.voxelDims+0.5*self.buffer)
            axFiber.set_ylim(0.5*self.buffer,self.voxelDims+0.5*self.buffer)
            axFiber.set_zlim(0.5*self.buffer,self.voxelDims+0.5*self.buffer)
            axFiber.set_xlabel(r'x')
            axFiber.set_ylabel(r'y')
            axFiber.set_zlabel(r'z')
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
            axExtra.set_xlabel(r'x')
            axExtra.set_ylabel(r'y')
            axExtra.set_zlabel(r'z')
            axExtra.legend()

        if any([plotFibers, plotExtra, plotCells]):  
            plt.show()
        return 

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

    def _signal_from_trajectory_data(self,trajectory_dir):
        Start = time.time()
        traj1s = []
        traj2s = []
        print(trajectory_dir)
        trajectory_t1ms = glob.glob(trajectory_dir + os.sep + '*T1m*.npy')
        for trajectory_file in trajectory_t1ms:
                traj_dir, fname = os.path.split(trajectory_file)
                compartment = (fname[0:5])
                traj1 = np.load(trajectory_file)
                traj2 = np.load(trajectory_file.replace('T1m', 'T2p'))

                traj1s.append(traj1)
                traj2s.append(traj2)
        if len(traj1s) == 3:
            traj1_combined = np.vstack([traj1s[0], traj1s[1], traj1s[2]])
            traj2_combined = np.vstack([traj2s[0], traj2s[1], traj2s[2]])
        elif len(traj1s) == 2:
            traj1_combined = np.vstack([traj1s[0], traj1s[1]])
            traj2_combined = np.vstack([traj2s[0], traj2s[1]])
    
        signal, bvals = self.signal(traj1_combined, traj2_combined, xyz = False, finite = True)
        dwi = nb.Nifti1Image(signal.reshape(1,1,1,-1), affine = np.eye(4))
        nb.save(dwi, traj_dir + os.sep  + "R=" + str(trajectory_dir[14]) + "_C=" + str(trajectory_dir[50]) + "_total_Signal_DBSI.nii")
        End = time.time()
        print('Trajectory Time: {} seconds'.format(End - Start))
        return

def dmri_sim_wrapper(arg):
    path, file = os.path.split(arg)
    x = dmri_simulation()
    x.from_config(arg)
    #x._set_params_from_config(arg)
    #x._signal_from_trajectory_data(path)

def main():       
    configs = glob.glob(r"C:\MCSIM\ISRM\**\*.ini")
    #print(configs)
    for cfg in configs:
        p = Process(target=dmri_sim_wrapper, args = (cfg,))
        p.start()
        p.join()

if __name__ == "__main__":
    main()

    
 



