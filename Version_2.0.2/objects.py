import numpy as np



class fiber():
    def __init__(self, center: np.ndarray, direction: np.ndarray, bundle: int, diffusivity: float, radius: float) -> None:
                
        self.center = center
        self.bundle = bundle + 1
        self.direction = direction
        self.diffusivity = diffusivity
        self.radius      = radius
        return

    """ Getters """
        
    def _get_center(self):
        return self.center
    
    def _get_bundle(self):
        return self.bundle
    
    def _get_direction(self):
        return self.direction
    
    def _get_diffusivity(self):
        return self.diffusivity
    
    def _get_radius(self):
        return self.radius
    
    



class cell():
    def __init__(self, cell_center, cell_radius: float, cell_diffusivity: float) -> None:
                
        self.center      = cell_center
        self.diffusivity = cell_diffusivity
        self.radius      = cell_radius

    """ Setters """     
    def _set_center(self, cell_center: np.ndarray):
        self.center = cell_center
        return
    
    def _set_diffusivity(self, D0: float):
        self.diffusivity = D0
        return
    
    def _set_radius(self, radius: float):
        self.radius = radius
        return
    
    """ Getters """
        
    def _get_center(self):
        return self.center
    
    def _get_diffusivity(self):
        return self.diffusivity
    
    def _get_radius(self):
        return self.radius
    
    
class spin():
    def __init__(self, spin_position_t1m : np.ndarray) -> None:

        self.position_t1m = spin_position_t1m
        self.position_t2p = np.empty(shape=(3,), dtype=np.float32)
        self.in_fiber_index = None
        self.fiber_bundle = None
        self.in_cell_index = None
        self.in_water_index = None
        
        return 
    
    def _set_fiber_index(self, index : int):
        if index < 0:
            return
        else:
            self.in_fiber_index = index
        return
    
    def _set_fiber_bundle(self, index : int): # call after _set_fiber_index
        if self.in_fiber_index != None:
            self.fiber_bundle = index
            return
        else:
            return

    def _set_position_t2p(self, position : np.ndarray):
        self.position_t2p = position
        return
    
    def _set_cell_index(self, index : int): # don't call this before _set_fiber_index ! 
        if np.logical_or(index < 0, self.in_fiber_index != None):
            self.in_cell_index = -1
            return
        else:
            self.in_cell_index = index
        return
    def _set_water_index(self, index : int):
        if index < 0:
            self.in_water_index = -1
            return 
        else:
            self.in_water_index = index
        return
            
    
    def _get_position_t1m(self):
        return self.position_t1m
    
    def _get_position_t2p(self):
        return self.position_t2p
    
    def _get_fiber_index(self):
        return self.in_fiber_index
    
    def _get_bundle_index(self):
        return self.fiber_bundle
    
    def _get_cell_index(self):
        return self.in_cell_index
    
    def _get_water_index(self):
        return self.in_water_index
    




