from typing import Dict, Union

import numpy as np



class fiber():
    """Class object for fiber attributes
    """  
    def __init__(self, 
                 center: np.ndarray, 
                 theta : np.ndarray,
                 direction: np.ndarray, 
                 bundle: int, 
                 step: float, 
                 radius: float,
                 kappa : float,
                 L     : float,
                 A     : float,
                 P     : float,
                 ) -> None:
        """Fiber information and parameters
        """ 

        self.center      = center
        self.theta       = theta
        self.bundle      = bundle + 1
        self.direction   = direction
        self.step        = step
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
    
  
class cell():
    def __init__(self, cell_center, cell_radius: float, cell_bundle : int) -> None:
        """Cell information

        """  
        self.center      = cell_center
        self.radius      = cell_radius
        self.bundle      = cell_bundle + 1
        return

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
        self.r0 = spin_position_t1m
        return 
    