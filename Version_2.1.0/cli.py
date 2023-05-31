import sys
import argparse
import simulation

def add_subgparser_args(subparsers: argparse) -> argparse:
    
    subparser = subparsers.add_parser("simulate",
                                      description="Simulate PGSE experiment"
                                      "on custom defined biological domain",
                                      )

    """  Simulation Parameters """

    subparser.add_argument("--n_walkers", nargs=None, type=int,
                           dest='n_walkers', default= 1e6,
                           required=False,
                           help="The number of spins to populate within the voxel, entered as an integer value. To obtain reliable results, use enough spins to acheive a minimal spin volume density of 1 per cubic micron.")
    subparser.add_argument("--fiber_fractions", nargs=None, type=str,
                  dest='fiber_fractions', default='0.2,   0.2',
                  required=False,
                  help="The desired volume fraction of each fiber type within its region of the voxel, entered as a comma-separated string of values between 0 and 1 (e.g., ''0.5, 0.7'')")

    subparser.add_argument("--fiber_radii", nargs=None, type=str,
                  dest='fiber_radii', default='1.5, 1.5',
                  required=False,
                  help="The radii (in units of micrometers) of each fiber type, entered as a comma-separated string (e.g., ''1.5, 2.0'')")

    subparser.add_argument("--thetas_Y", nargs=None, type=str,
                  dest='thetas', default='0, 60',
                  required=False,
                  help="Rotation angle (with respect to the Y axis, in degrees) for each fiber type, entered as a comma-separated string (e.g., ''0, 30'')")

    subparser.add_argument("--fiber_diffusions", nargs=None, type=str,
                  dest='fiber_diffusions', default='1.0, 2.0',
                  required=False,
                  help="Diffusivity within each fiber type (in units of micrometers^2 per ms), entered as a comma-separated string (e.g., ''1.0, 2.5'')")

    subparser.add_argument("--cell_fractions", nargs=None, type=str,
                  dest='cell_fractions', default='0.1, 0.0',
                  required=False,
                  help="The desired volume fraction of each cell type, entered as a comma-separated string of values between 0 and 1 (e.g., ''0.05, 0.20'')")

    subparser.add_argument("--cell_radii", nargs=None, type=str,
                  dest='cell_radii', default='10.0, 10.0',
                  required=False,
                  help="The radii (in units of micrometers) of each cell type, entered as a comma-separated string (e.g., ''3.0, 7.5'')")

    subparser.add_argument("--fiber_configuration", nargs=None, type=str,
                  dest='fiber_configuration', default='Penetrating',
                  required=False,
                  help="Fiber configuration/geometry type. See README for addition details.")


    subparser.add_argument("--water_diffusivity", nargs=None, type=float,
                  dest='water_diffusivity', default=3.0,
                  required=False,
                  help="Diffusivity of free water (in units of micrometers^2 per ms)")
    

    """  Scanning Parameters """

    subparser.add_argument("--Delta", nargs=None, type=float,
                           dest='Delta', default=0.001,
                           required=False,
                           help="Diffusion time, in units of milliseconds"
                           )

    subparser.add_argument("--dt", nargs=None, type=float,
                           dest='dt', default=0.001,
                           required=False,
                           help="Time step for simulation, in units of milliseconds. Do not alter from default value of 0.001 ms."
                           )

    subparser.add_argument("--voxel_dims", nargs=None, type=float,
                           dest='voxel_dims', default=75.,
                           required=False,
                           help="Length of isotropic voxel, in units of micrometers. (Note: Currently, only isotropic voxels are supported.)"
                           )

    subparser.add_argument("--buffer", nargs=None, type=float,
                           dest='buffer', default=10.,
                           required=False,
                           help="Additional length, in micrometers, added to each voxel dimension when populating cells and fibers. It is not recommended for buffers to exceed 15 percent of the voxel size."
                           )

    subparser.add_argument("--bvals", nargs=None, type=str,
                           dest='input_bvals', default=None,
                           required=False,
                           help="Gradient magnitudes (b values) used for diffusion MR signal generation."
                           )

    subparser.add_argument("--bvecs", nargs=None, type=str,
                           dest='input_bvecs', default=None,
                           required=False,
                           help="Gradient directions (b vectors) used  for diffusion MR signal generation."
                           )
    
    subparser.add_argument("--void_dist", nargs = None, type = float,
                           dest = 'void_dist', default = 0,
                           required = False,
                           help = "Size (in units of micrometers) of a region in middle of voxel, aka void, excluded from fiber placement. (optional except when fiber_configuration = ''Void'') "
                           )

    return subparsers

def typing_and_validation(args):
    # N_walkers
    args['n_walkers'] = int(args['n_walkers'])
    if args['n_walkers'] <= 0:
        raise ValueError(f"Entered: {args['n_walkers']}, but N_walkers must be positive")
    # Fiber-Fractions
    args['fiber_fractions'] = [float(frac) for frac in str(args['fiber_fractions']).split(',')]
    if any([ff < 0 for ff in args['fiber_fractions']]) or sum(args['fiber_fractions']) > 1:
        raise ValueError(f"Entered: {args['fiber_fractions']}, but each fiber fraction must be non-negative and cannot sum to more than 1.0")
    # Fiber-Radii
    args['fiber_radii'] = [float(rad) for rad in str(args['fiber_radii']).split(',')]
    if any([radius < 0 for radius in args['fiber_radii']]):
        raise ValueError(f"Entered: {args['fiber_radii']}, but fiber radii must be non-negative")
    # Thetas
    args['thetas'] = [float(theta) for theta in str(args['thetas']).split(',')]
    # Fiber Diffusions
    args['fiber_diffusions'] = [float(D0) for D0 in str(args['fiber_diffusions']).split(',')]
    if any([D0 < 0 for D0 in args['fiber_diffusions']]):
        raise ValueError(f"Entered: {args['fiber_diffusions']}, but each fiber diffusion must be non-negative")
    # Cell Fractions
    args['cell_fractions'] = [float(frac) for frac in str(args['cell_fractions']).split(',')]
    if any(cf < 0 for cf in args['cell_fractions']) or sum(args['cell_fractions']) > 1:
        raise ValueError(f"Entered: {args['cell_fractions']}, but each cell fraction must be non-negative and cannot sum to more than 1.0")
    # Cell Radii 
    args['cell_radii'] = [float(rad) for rad in str(args['cell_radii']).split(',')]
    if any(cr < 0 for cr in args['cell_radii']):
        raise ValueError(f"Entered: {args['cell_radii']}, but each cell radius must be non-negative")
    # Water Diffusivity
    args['water_diffusivity'] = float(args['water_diffusivity'])
    if args['water_diffusivity'] <= 0:
        raise ValueError(f"Entered {args['water_diffusivity']}, but water diffusivity must be non-negative")
    # Delta
    args['Delta'] = float(args['Delta'])
    if args['Delta'] < 0:
        raise ValueError(f"Entered: {args['Delta']}, but Delta must be positive")
    # delta / dt
    args['dt'] = float(args['dt'])
    if args['dt'] <= 0:
        raise ValueError(f"Entered: {args['dt']}, but dt must be positive")
    # Voxel Dims
    args['voxel_dims'] = float(args['voxel_dims'])
    if args['voxel_dims'] <= 0:
        raise ValueError(f"Entered: {args['voxel_dims']}, but voxel_dims must be positive")
    # Void Distance
    args['void_dist'] = float(args['void_dist'])
    if args['void_dist'] < 0:
        raise ValueError(f"Entered: {args['void_dist']}, but void_dist must be non-negative")
    return args

def main():

    parser = argparse.ArgumentParser(prog='cli',
                                     description='What the program does',
                                     epilog='Text at the bottom of help')

    subparsers = parser.add_subparsers(help='sub-command help')
    subparsers = add_subgparser_args(subparsers)
    args = vars(parser.parse_args())

    args = typing_and_validation(args)

    simulation.run(args)

if __name__ == "__main__":
    main()
