import matplotlib.pyplot as plt
import numpy as np 


def plot_axon(Data, fiber_radius, voxel_dims):
    x_center = Data[0]
    y_center = Data[1]
    height = 8
    radius = fiber_radius
    z = np.linspace(-10, np.amax(voxel_dims) + 10, 2)
    theta = np.linspace(0,2*np.pi, 25)
    th, zs = np.meshgrid(theta,z)
    xs = radius * np.cos(th) + x_center
    ys = radius * np.sin(th) + y_center
    return xs, ys, zs


 
def plot_cell(center):
    xc = center[0]
    yc = center[1]
    zc = center[2]
    cell_radius = center[3]
    r = cell_radius
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = xc + r*np.cos(u) * np.sin(v)
    y = yc + r*np.sin(u) * np.sin(v)
    z = zc + r*np.cos(v)
    return x, y, z

def plot(fiber_xycordinate, cell_centers, Spins, spin_loc_key, spin_trajectory, voxel_dims):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(projection = '3d')
    colors = ['r', 'g', 'b']
   
    if (len(fiber_xycordinate.shape)) > 1:
   
        for i in range(len(fiber_xycordinate[:,0])):
            xs, ys, zs = plot_axon(fiber_xycordinate[i,:], fiber_xycordinate[i,5], voxel_dims)

            if(fiber_xycordinate[i,6] != 0):
                if(fiber_xycordinate[i,3]) == 0:
                    ax.plot_surface(xs, ys, zs, color = colors[int(fiber_xycordinate[i,6])-1], alpha = .25)
                    z = 0
                elif (fiber_xycordinate[i,3]) == 1:
                    ax.plot_surface(zs, ys, xs, color = colors[int(fiber_xycordinate[i,6])-1], alpha = .25)
                    z = 0

    else:
        for i in range(len(fiber_xycordinate)):
            xs, ys, zs = plot_axon(fiber_xycordinate, fiber_xycordinate[5], voxel_dims = np.array([0,200]))
            if(fiber_xycordinate[6] != 0):
                if(fiber_xycordinate[3]) == 0:
                    ax.plot_surface(xs, ys, zs, color = colors[int(fiber_xycordinate[6])-1], alpha = .25)
                elif (fiber_xycordinate[3]) == 1:
                    ax.plot_surface(zs, ys, xs, color = colors[int(fiber_xycordinate[6])-1], alpha = .25)
    if (len(cell_centers.shape)) > 1:
        for i in range(len(cell_centers[:,0])):
            xs, ys, zs = plot_cell(cell_centers[i,:])
            ax.plot_surface(xs, ys,zs, cmap = 'viridis', alpha = .50)
    else:
        xs, ys, zs = plot_cell(cell_centers)
        ax.plot_surface(xs, ys, zs, cmap = 'viridis', alpha = .50)


    color = ['aqua', 'orange', 'purple']
    for i in range(Spins.shape[0]):
        if  spin_loc_key[i] >= 1:
            ax.scatter3D(spin_trajectory[i,:,0], spin_trajectory[i,:,1], spin_trajectory[i,:,2], s = 1, color = 'black')
       

    ax.set_xlabel(r'$x \quad \mu m$')
    ax.set_ylabel(r'$y \quad \mu m$')
    ax.set_zlabel(r'$z \quad \mu m$')
    ax.set_title(r'Simulation Geometry')

    ax.set_ylim(0,200)
    ax.set_xlim(0,200)

    #ax.view_init(0,180)
    plt.show()
    #plt.savefig(r"C:\temp\MCSIM_geometry_45.png")
