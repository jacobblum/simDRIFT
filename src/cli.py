import sys
import argparse
import src.simulation as simulation
from pathlib import Path
import numba
import os 

class CLI:
    def __init__(self, subparsers) -> None:
        self.subparsers = subparsers        
        pass

    def validate_args(self, args):     
        # N_walkers
        args['n_walkers'] = int(args['n_walkers'])
        #assert args['n_walkers']/(args['voxel_dims']**3) > 1.0," --Simulation requires spin densities > 1.0 per cubic micron"
        
        # Fiber-Fractions
        args['fiber_fractions'] = [float(frac) for frac in str(args['fiber_fractions']).split(',')]
        for ff in args['fiber_fractions']:
            assert ff >= 0, "--fiber_fractions must be non-negative"
            assert ff < 1, "--fiber_fractions must be less than 1.0"
        assert sum(args['fiber_fractions']) < 1, "--fiber_fractions cannot sum to more than 1.0"
        
        # Fiber-Radii
        args['fiber_radii'] = [float(rad) for rad in str(args['fiber_radii']).split(',')]
        for radius in args['fiber_radii']:
            assert radius >= 0, "--fiber_radii must be non-negative"
        
        # Thetas
        args['thetas'] = [float(theta) for theta in str(args['thetas']).split(',')]
        
        # Fiber Diffusions
        args['fiber_diffusions'] = [float(D0) for D0 in str(args['fiber_diffusions']).split(',')]
        for D0 in args['fiber_diffusions']:
            assert D0 >= 0, "--fiber_diffusions must be non-negative"
        
        # Cell Fractions
        args['cell_fractions'] = [float(fraction) for fraction in str(args['cell_fractions']).split(',')]
        for cf in args['cell_fractions']:
            assert cf >= 0, "--cell_fractions must be non-negative"
            assert cf < 1, "--cell_fractions must be less than 1.0"
        assert sum(args['cell_fractions']) < 1, "--cell_fractions cannot sum to more than 1.0"
        
        # Cell Radii 
        args['cell_radii'] = [float(rad) for rad in str(args['cell_radii']).split(',')]
        for cr in args['cell_radii']:
            assert cr >= 0, "--cell_radii must be non-negative"
        
        # Water Diffusivity
        args['water_diffusivity'] = float(args['water_diffusivity'])
        assert args['water_diffusivity'] >= 0.0, "--water_diffusivity must be non-negative"
        
        # Delta
        args['Delta'] = float(args['Delta'])
        assert args['Delta'] > 0, "--Delta must be positive"
        
        # delta / dt
        args['dt'] = float(args['dt'])
        assert args['dt'] > 0, "--dt must be positive"
        
        # Voxel Dims
        args['voxel_dims'] = float(args['voxel_dims'])
        assert args['voxel_dims'] > 0, "--voxel dims must be positive"
        
        # Void Distance
        args['void_dist'] = float(args['void_dist'])
        assert args['void_dist'] >= 0, "--void_dist must be non-negative"

        assert args['verbose'] == 'yes' or args['verbose'] == 'no', "--verbose must be yes or no"

        ## Check that GPU is available 
        assert numba.cuda.is_available(), "Trying to use Cuda device, " \
                                        "but Cuda device is not available."
        
        ## If using custom diffusion scheme... make sure that the bval and bvec paths exist
        if all([args['input_bvals'], args['input_bvecs']]):
            args['CUSTOM_DIFF_SCHEME_FLAG'] = True
            assert all([os.path.exists(path) for path in [args['input_bvals'], args['input_bvecs']]])
        else:
            args['CUSTOM_DIFF_SCHEME_FLAG'] = False        

        return args
    
    def run(self, args):
        simulation.run(args)


    def add_subparser_args(self) -> argparse:
        
        subparser = self.subparsers.add_parser("simulate",
                                        description="Simulate PGSE experiment"
                                        "on custom defined biological domain",
                                        )

        """  Simulation Parameters """

        subparser.add_argument("--n_walkers", nargs=None, type=int,
                            dest='n_walkers', default= 1e6,
                            required=False,
                            help="The number of spins to populate within the voxel, entered as an integer value. To obtain reliable results, use enough spins to acheive a minimal spin volume density of 1 per cubic micron.")
        subparser.add_argument("--fiber_fractions", nargs=None, type=str,
                    dest='fiber_fractions', default='0.5, 0.3, 0.19',
                    required=False,
                    help="The desired volume fraction of each fiber type within its region of the voxel, entered as a comma-separated string of values between 0 and 1 (e.g., ''0.5, 0.7'')")

        subparser.add_argument("--fiber_radii", nargs=None, type=str,
                    dest='fiber_radii', default='1.0, 1.0, 1.0',
                    required=False,
                    help="The radii (in units of micrometers) of each fiber type, entered as a comma-separated string (e.g., ''1.5, 2.0'')")

        subparser.add_argument("--thetas", nargs=None, type=str,
                    dest='thetas', default='0, 0,0',
                    required=False,
                    help="Rotation angle (with respect to the Y axis, in degrees) for each fiber type, entered as a comma-separated string (e.g., ''0, 30'')")

        subparser.add_argument("--fiber_diffusions", nargs=None, type=str,
                    dest='fiber_diffusions', default='1.0,2.0,1.5',
                    required=False,
                    help="Diffusivity within each fiber type (in units of micrometers^2 per ms), entered as a comma-separated string (e.g., ''1.0, 2.5'')")

        subparser.add_argument("--cell_fractions", nargs=None, type=str,
                    dest='cell_fractions', default='0.0, 0.0',
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
                            dest='buffer', default=0.,
                            required=False,
                            help="Additional length, in micrometers, added to each voxel dimension when populating cells and fibers. It is not recommended for buffers to exceed 15 percent of the voxel size."
                            )

        subparser.add_argument("--bvals", nargs=None, type=str,
                            dest='input_bvals', default=None,
                            required=False,
                            help="Path to file containing gradient magnitudes (b values) used for diffusion MR signal generation."
                            )

        subparser.add_argument("--bvecs", nargs=None, type=str,
                            dest='input_bvecs', default=None,
                            required=False,
                            help="Path to file containing gradient directions (b vectors) used for diffusion MR signal generation."
                            )
        
        subparser.add_argument("--void_dist", nargs = None, type = float,
                            dest = 'void_dist', default = 0,
                            required = False,
                            help = "Size (in units of micrometers) of a region in middle of voxel, aka void, excluded from fiber placement. (optional except when fiber_configuration = ''Void'') "
                            )
        
        subparser.add_argument("--diff_scheme", nargs=None, type=str,
                            dest='diff_scheme', default='DBSI_99',
                            required=False,
                            help="diffusion scheme."
                            )

        subparser.add_argument("--verbose", nargs = None, type = str,
                            dest = 'verbose', default = 'yes',
                            required = False,
                            help = "print logging information"
                            )

        return self.subparsers

#def _make_args_dict():
#    parser = argparse.ArgumentParser(prog='cli',
#                                    description='What the program does',
#                                    epilog='Text at the bottom of help')
#
#    subparsers = parser.add_subparsers(help='sub-command help')
#    subparsers = add_subgparser_args(subparsers)
#    args = vars(parser.parse_args())
#    args = typing_and_validation(args)
#
#    return args
