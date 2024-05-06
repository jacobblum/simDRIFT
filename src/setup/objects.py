from typing import Dict, Union

import numpy as np



class fiber():
    """Class object for fiber attributes
    """  
    def __init__(self, 
                 center: np.ndarray, 
                 direction: np.ndarray, 
                 bundle: int, 
                 diffusivity: float, 
                 radius: float,
                 kappa : float,
                 L     : float,
                 A     : float,
                 P     : float
                 ) -> None:
        """Fiber information and parameters

        :param center: (x,y,z) coordinates of the fiber center
        :type center: np.ndarray
        :param direction: Constituent bundle index
        :type direction: np.ndarray
        :param bundle: Unit vector pointed along the fiber direction
        :type bundle: int
        :param diffusivity: Intrinsic fiber diffusivity, square micrometers per millisecond
        :type diffusivity: float
        :param radius: Fiber radius, in micrometers
        :type radius: float
        """ 

        self.center      = center
        self.bundle      = bundle + 1
        self.direction   = direction
        self.diffusivity = diffusivity
        self.radius      = radius
        

        # ------------------------------------------------------------------------------- #
        #                                Fiber Bending                                    #
        # ------------------------------------------------------------------------------- #
        self.kappa = kappa
        self.L     = L
        self.A     = A
        self.P     = P
        return
    
    def _gamma(self, r: np.ndarray) -> np.ndarray:
        t = np.einsum('i,i', r, self.direction)
        return np.array([self.A*np.sin(np.pi * self.kappa / ( (1/self.P) * self.L) * t),0, t])
    
    def _d_gamma__d_t(self, r: np.ndarray) -> np.ndarray:
        t = np.einsum('i,i', r, self.direction)
        gamma_prime = np.array([self.A *np.pi * self.kappa / ( (1/self.P) * self.L) * np.cos(np.pi * self.kappa / ( (1/self.P) * self.L) * t), 0 ,1])
        gamma_prime /= np.linalg.norm(gamma_prime, ord = 2)
        return gamma_prime
    
    @property
    def theta(self):
        return np.arccos(np.einsum('i,i', np.array([0., 0., 1.0]), self.direction))
 
    """ Getters """
        
    def _get_center(self):
        """Returns an array containing coordinates for the center of the specified fiber

        :return: An array of fiber center coordinates
        :rtype: numpy.ndarray
        """  
        return self.center
    
    def _get_bundle(self):
        """Returns the fiber bundle index for the specified fiber

        :return: The bundle/type index for the specified fiber
        :rtype: int
        """
        return self.bundle
    
    def _get_direction(self):
        """Returns the orientation vector for the specified fiber

        :return: The orientation vector/rotation matrix for the specified fiber
        :rtype: numpy.ndarray
        """
        return self.direction
    
    def _get_diffusivity(self):
        """Returns the intra-axonal diffusivity for the specified fiber, in units of :math:`{\mathrm{μm}^2}\\, \mathrm{ms}^{-1}`

        :return: The user-specified diffusivity within the specified fiber
        :rtype: float
        """
        return self.diffusivity
    
    def _get_radius(self):
        """Returns the radius of the specified fiber, in units of :math:`{\mathrm{μm}}`

        :return: The user-specified radius of the specified fiber
        :rtype: float
        """
        return self.radius
    
    



class cell():
    def __init__(self, cell_center, cell_radius: float, cell_diffusivity: float) -> None:
        """Cell information

        :param cell_center: (x,y,z) coordinates of the cell center 
        :type cell_center: np.ndarray
        :param cell_radius: Intrinsic fiber diffusivity
        :type cell_radius: float
        :param cell_diffusivity: Fiber radius
        :type cell_diffusivity: float
        """
                
        self.center      = cell_center
        self.diffusivity = cell_diffusivity
        self.radius      = cell_radius

    """ Setters """     
    def _set_center(self, cell_center: np.ndarray):
        """Records the center coordinates for the specified cell

        :param cell_center: An array containing the cell center coordinates
        :type cell_center: np.ndarray
        """        
        self.center = cell_center
        return
    
    def _set_diffusivity(self, D0: float):
        """Records the diffusivity within the specified cell in units of :math:`{\mathrm{μm}^2}\\, \mathrm{ms}^{-1}`

        :param D0: The diffusivity within the cell, assumed to be equal to the user-specified free water diffusivity
        :type D0: float
        """     
        self.diffusivity = D0
        return
    
    def _set_radius(self, radius: float):
        """Records the radius of the specified cell, in units of :math:`{\mathrm{μm}}`

        :param radius: The cell radius
        :type radius: float
        """        
        self.radius = radius
        return
    
    """ Getters """
        
    def _get_center(self):
        """Returns an array containing coordinates for the center of the specified cell

        :return: An array containing the cell center coordinates
        :rtype: numpy.ndarray
        """    
        return self.center
    
    def _get_diffusivity(self):
        """Returns the diffusivity within the specified cell in units of :math:`{\mathrm{μm}^2}\\, \mathrm{ms}^{-1}`

        :return: The diffusivity within the cell
        :rtype: float
        """    
        return self.diffusivity
    
    def _get_radius(self):
        """Returns the radius of the specified cell, in units of :math:`{\mathrm{μm}}`

        :return: The cell radius
        :rtype: float
        """   
        return self.radius
    
    
class spin():
    def __init__(self, spin_position_t1m : np.ndarray) -> None:
        """Spin information

        :param spin_position_t1m: Initial spin position
        :type spin_position_t1m: np.ndarray
        :param spin_position_t1m: Final spin position
        :type spin_position_t1m: np.ndarray
        :param in_fiber_index: Index of resident fiber (if the spin resides in a fiber)
        :type in_fiber_index: int
        :param fiber_bundle: Index of resident fiber bundle (if the spin resides in a fiber)
        :type fiber_bundle: int
        :param in_cell_index: Index of the resident cell (if the spin resides in a cell) 
        :type in_cell_index: int
        :param in_water_index: Spin index if in water
        :type in_water_index: int
        """

        self.position_t1m = spin_position_t1m
        self.position_t2p = np.empty(shape=(3,), dtype=np.float32)
        self.in_fiber_index = None
        self.fiber_bundle   = None
        self.in_cell_index  = None
        self.in_water_index = None
        
        return 
    
    def _set_fiber_index(self, index : int):
        """Records the index of the fiber in which the spin resides.

        :param index: Index of resident fiber (if the spin resides in a fiber)
        :type index: np.ndarray
        """        
        if index < 0:
            return
        else:
            self.in_fiber_index = index
        return
    
    def _set_fiber_bundle(self, index : int): # call after _set_fiber_index
        """Records the fiber bundle for a spin residing in a fiber 

        :param index: Index of the resident fiber bundle (if the spin resides in a fiber) 
        :type index: int
        """     
        if self.in_fiber_index != None:
            self.fiber_bundle = index
            return
        else:
            return

    def _set_position_t2p(self, position : np.ndarray):
        """Records the position of the spin after diffusion time :math:`\Delta`.

        :param position: Final spin position 
        :type position: np.ndarray
        """
        self.position_t2p = position
        return
    
    def _set_cell_index(self, index : int): # don't call this before _set_fiber_index! 
        """Records the index of the spin if it resides in a cell 

        :param index: Index of the resident cell (if the spin resides in a cell) 
        :type index: int
        """        
        if np.logical_or(index < 0, self.in_fiber_index != None):
            self.in_cell_index = -1
            return
        else:
            self.in_cell_index = index
        return
    def _set_water_index(self, index : int):
        """Records the index of the spin if it resides in water 

        :param index: Spin index if in water
        :type index: int
        """             
        if index < 0:
            self.in_water_index = -1
            return 
        else:
            self.in_water_index = index
        return
            
    
    def _get_position_t1m(self):
        """Returns the initial position of the spin

        :return: Initial spin position
        :rtype: int
        """        
        return self.position_t1m
    
    def _get_position_t2p(self):
        """Returns the final position of the spin

        :return: Final spin position
        :rtype: int
        """        
        return self.position_t2p
    
    def _get_fiber_index(self):
        """Returns the index of the fiber in which the spin resides.

        :return: Index of the resident fiber bundle (if the spin resides in a fiber) 
        :rtype: int
        """        
        return self.in_fiber_index
    
    def _get_bundle_index(self):
        """Returns the fiber bundle index for a spin residing in a fiber 

        :return: Index of the resident fiber bundle (if the spin resides in a fiber) 
        :rtype: int
        """        
        return self.fiber_bundle
    
    def _get_cell_index(self):
        """Returns the index of the spin if it resides in a cell 

        :return: Index of the resident cell (if the spin resides in a cell) 
        :rtype: int
        """        
        return self.in_cell_index
    
    def _get_water_index(self):
        """Returns the water index of the specified spin

        :return: Spin index if in water
        :rtype: int
        """        
        return self.in_water_index
    




