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
                           dest='n_walkers', default= 1000 * 1e3,
                           required=False,
                           help="Please enter the relative"
                           "path for the diffusion data file on which"
                           "to run NEO as a .nii or .nii.gz file"
                           )
    subparser.add_argument("--fiber_fractions", nargs=None, type=str,
                  dest='fiber_fractions', default='0.5, 0.5',
                  required=False,
                  help="The volume fractions of each of the fiber bundles"
                  "separated by a comma")

    subparser.add_argument("--fiber_radii", nargs=None, type=str,
                  dest='fiber_radii', default='1.0, 1.0',
                  required=False,
                  help="The Fiber_Radii")

    subparser.add_argument("--thetas", nargs=None, type=str,
                  dest='thetas', default='0, 90',
                  required=False,
                  help="The Fiber Orientations")

    subparser.add_argument("--fiber_diffusions", nargs=None, type=str,
                  dest='fiber_diffusions', default='1.0, 2.0',
                  required=False,
                  help="The Fiber Diffusions")

    subparser.add_argument("--cell_fractions", nargs=None, type=str,
                  dest='cell_fractions', default='0.1, 0.1',
                  required=False,
                  help="The Volume of the Cells")

    subparser.add_argument("--cell_radii", nargs=None, type=str,
                  dest='cell_radii', default='5., 5.',
                  required=False,
                  help="The Radii of the Cells")

    subparser.add_argument("--fiber_configuration", nargs=None, type=str,
                  dest='fiber_configuration', default='Penetrating',
                  required=False,
                  help="The Fiber Configuration")

    subparser.add_argument("--simulate_fibers", nargs=None, type=bool,
                  dest='simulate_fibers', default=True,
                  required=False,
                  help="Simulate the Fibers ? ")

    subparser.add_argument("--simulate_cells", nargs=None, type=bool,
                  dest='simulate_cells', default=True,
                  required=False,
                  help="Simulate the Cells ? ")

    subparser.add_argument("--simulate_water", nargs=None, type=bool,
                  dest='simulate_water', default=True,
                  required=False,
                  help="Simulate the Water ? ")

    """  Scanning Parameters """

    subparser.add_argument("--Delta", nargs=None, type=float,
                           dest='Delta', default=0.10,
                           required=False,
                           help="The Diffusion Time"
                           )

    subparser.add_argument("--dt", nargs=None, type=float,
                           dest='dt', default=0.001,
                           required=False,
                           help="Time Discretization Factor"
                           )

    subparser.add_argument("--voxel_dims", nargs=None, type=float,
                           dest='voxel_dims', default=50.,
                           required=False,
                           help="Voxel Size"
                           )

    subparser.add_argument("--buffer", nargs=None, type=float,
                           dest='buffer', default=5.,
                           required=False,
                           help="Voxel Buffer"
                           )

    subparser.add_argument("--bvals", nargs=None, type=str,
                           dest='input_bvals', default=None,
                           required=False,
                           help="bvalues used in the diffusion experiment"
                           )

    subparser.add_argument("--bvecs", nargs=None, type=str,
                           dest='input_bvecs', default=None,
                           required=False,
                           help="b-vectors used in the diffusion experiment"
                           )
    
    subparser.add_argument("--void_dist", nargs = None, type = float,
                           dest = 'void_dist', default = 0.,
                           required = False,
                           help = "void distance for void configuration"
                           )

    return subparsers

def typing(args):

    args['n_walkers'] = int(args['n_walkers'])
    args['fiber_fractions'] = [float(frac) for frac in str(args['fiber_fractions']).split(',')]
    args['fiber_radii'] = [float(rad) for rad in str(args['fiber_radii']).split(',')]
    args['thetas'] = [float(theta) for theta in str(args['thetas']).split(',')]
    args['fiber_diffusions'] = [float(D0) for D0 in str(args['fiber_diffusions']).split(',')]
    args['cell_fractions'] = [float(frac) for frac in str(args['cell_fractions']).split(',')]
    args['cell_radii'] = [float(rad) for rad in str(args['cell_radii']).split(',')]
    args['Delta'] = float(args['Delta'])
    args['dt'] = float(args['dt'])
    args['voxel_dims'] = float(args['voxel_dims'])
    args['void_dist'] = float(args['void_dist'])
    return args


def main():

    parser = argparse.ArgumentParser(prog='cli',
                                     description='What the program does',
                                     epilog='Text at the bottom of help')

    subparsers = parser.add_subparsers(help='sub-command help')
    subparsers = add_subgparser_args(subparsers)
    args = vars(parser.parse_args())

    args = typing(args)

    simulation.run(args)

if __name__ == "__main__":
    main()
