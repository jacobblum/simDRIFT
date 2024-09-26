import torch

def _axis_oriented_distance(x : torch.FloatTensor, y : torch.FloatTensor, v : torch.FloatTensor) -> torch.FloatTensor:
    """
    :param x: An (Nspins, 3) array of proposed spin positions
    :param y: An (Nspins, 3) array of the current (potentially dynamic) fiber center
    :param v: An (Nspins, 3) array of the current (potentially dynamic) fiber direction 
    
    return: The L2 Distance between x and y along the axis v
    """
    d = x - y

    d_l_2_sqr = torch.square(torch.linalg.norm(d, axis = -1, ord = 2)) 
    p_sqr     = torch.square(torch.einsum('Ni, Ni -> N', d, v))

    return torch.sqrt(d_l_2_sqr - p_sqr)

def _gamma_t(r, v, theta, kappa, L, A, P, force_broadcast : bool = False) -> torch.cuda.FloatTensor:
    # Native Call
    if (r.shape[0] == v.shape[0]) and (not force_broadcast):
        g_t = torch.zeros( 
                          size = (r.shape[0], 3), 
                          dtype= r.dtype,
                          device = r.device
                          )
        
        t = torch.einsum('Ni, Ni -> N', r, v) # Distance Along Principal Fiber Axis 
    # Broadcasted Call 
    elif (r.shape[0] != v.shape[0]) or (force_broadcast):
        g_t = torch.zeros( 
                          size  = (r.shape[0], v.shape[0], 3), 
                          dtype = r.dtype,
                          device = r.device
                          )
        t = torch.einsum('Mi, Ni -> MN', r, v)

    x = A * torch.sin(torch.pi * kappa /((1/P)*L)*t) 
    z = t 

    g_t[...,  0] =  torch.cos(theta)*x  + torch.sin(theta)*z
    g_t[..., -1] = -torch.sin(theta)*x  + torch.cos(theta)*z

    return g_t 

def _d_gt__d_t(r, v, theta, kappa, L, A, P, force_broadcast : bool = False) -> torch.cuda.FloatTensor:
  
    if (r.shape[0] == v.shape[0]) and (not force_broadcast):
        d_gt__d_t = torch.zeros( 
                          size = (r.shape[0], 3), 
                          dtype= r.dtype,
                          device = r.device
                          )
        
        t = torch.einsum('Ni, Ni -> N', r, v) # Distance Along Principal Fiber Axis 

    elif (r.shape[0] != v.shape[0]) or (force_broadcast):
        d_gt__d_t = torch.zeros( 
                          size  = (r.shape[0], v.shape[0], 3), 
                          dtype = r.dtype,
                          device = r.device
                          )
        t = torch.einsum('Mi, Ni -> MN', r, v)

    x = A*(torch.pi*kappa /((1/P)*L)) * torch.cos(torch.pi*kappa /((1/P)*L)*t)
    
    l_2_norm = torch.sqrt( torch.square(x) + 1)

    x /= l_2_norm
    z = 1/l_2_norm

    d_gt__d_t[...,  0] = torch.cos(theta)*x   + torch.sin(theta)*z
    d_gt__d_t[..., -1] = -torch.sin(theta)*x  + torch.cos(theta)*z

    return d_gt__d_t

