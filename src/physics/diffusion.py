import numpy as np 
import time
import sys
import logging
from typing import Dict, Type, Union, List
import torch 
import shutil
import subprocess
import psutil
import platform
import os 
from datetime import datetime
from joblib import Parallel, delayed 
import csv 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.physics.walks import (
    fiber_step,
    cell_step,
    water_step
)
from src.physics.common import (
    _axis_oriented_distance, 
    _gamma_t, 
    _d_gt__d_t
)

logger = logging.getLogger('simDRIFT')

class Namespace:
    def __init__(self, *args):
        self.__dict__.update(*args)

RANDOM_WAlK_DATA = [
    'use_cuda',
    'use_multiprocessing_diffusion',
    'n_cores',
    'Delta',
    'delta',
    'TE',
    'dt',
    'G',
    'fibers',
    'cells',
    'spins',
    'sim_out_dir',
    'D0'
]

OBJECT_TO_TORCH = [
    'fibers', 
    'spins', 
    'cells'
]

KEY_MAP_NAMES = [
    'ith_spin_in_jth_fiber_key', 
    'ith_spin_in_jth_cell_key', 
    'water_key'
]


GAMMA = 267.513e6 

def get_gpu_split_size(args):
    # Send all data to the gpu in a single block.
    return torch.full(
        (1, ), 
        fill_value=args.spins_r0.shape[0], 
        dtype=int, 
        device=args.spins_r0.device 
    )

def get_cpu_split_size(args):

    # get availalbe RAM [bytes]
    available_memory_bytes = psutil.virtual_memory().free
    MAX_PROCESS_MEMORY = 0.50 * (available_memory_bytes / args.n_cores)
    # calculate the approximate memory consumption by each process to be initiated
    f_frac = (args.ith_spin_in_jth_fiber_key > -1).sum() / args.spins_r0.shape[0] # fiber fraction
    c_frac = (args.ith_spin_in_jth_cell_key > -1).sum() / args.spins_r0.shape[0] # cell fraction
    w_frac = (args.water_key > -1).sum() / args.spins_r0.shape[0] # cell fraction

    MAX_MEM_SPLIT_SIZE = int( 
                             torch.sqrt( MAX_PROCESS_MEMORY / 4*(f_frac*23 + w_frac * 9 * args.fibers_center.shape[0] + c_frac * 3 * args.cells_center.shape[0] ) ).item()
                            )

    split_sizes = torch.full( (args.n_cores,), fill_value = args.spins_r0.shape[0] // args.n_cores, dtype= int )
    split_sizes[0:args.spins_r0.shape[0] % args.n_cores] += 1

    if split_sizes.max() >  MAX_MEM_SPLIT_SIZE:
        # floor(x/n) \leq x/n for all x,n in R^{+}, so this should be okay...
        split_sizes = torch.full( (args.spins_r0.shape[0] // MAX_MEM_SPLIT_SIZE, ), fill_value = MAX_MEM_SPLIT_SIZE, dtype = int) 
        # Distribute the Remainder as evenly as possible 
        split_sizes[:] += (args.spins_r0.shape[0] % (MAX_MEM_SPLIT_SIZE)) // (args.spins_r0.shape[0] // MAX_MEM_SPLIT_SIZE)
        # Put whatever is left into the last batch
        split_sizes[-1] = args.spins_r0.shape[0] - split_sizes[:-1].sum()

    return split_sizes


def get_gpu_stats() -> List[str]:
    gpu_query = r'name,timestamp,temperature.gpu,utilization.gpu,utilization.memory,memory.free,memory.used,memory.total'
    format = r'csv,nounits,noheader'
    result = subprocess.run(
                        [shutil.which("nvidia-smi"), f"--query-gpu={gpu_query}", f"--format={format}"],
                        encoding="utf-8",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,  
                        check=True,
                        ) 
    gpu_stats = result.stdout.strip().split(",")
    return gpu_stats

def get_cpu_stats() -> List[str]:
    process = psutil.Process(os.getpid())
    mem     = psutil.virtual_memory()
    cpu_query = {
                 'name' : platform.processor(), 
                 'timestamp': datetime.now().strftime("%m/%d/%Y, %H:%M:%S.%f")[:-3],
                 'temperature' : 0.,
                 'utilization.cpu' :    psutil.getloadavg()[0] / psutil.cpu_count() * 100,
                 'utilization.memory' : mem.percent,
                 'memory.free'  :       mem.free >> 20,
                 'memory.used'  :       np.round(process.memory_info().rss * 1e-6, 5),
                 'memory.total' :       mem.total >> 20
                }    
    cpu_stats = [str(v) for (k,v) in cpu_query.items()]
    return cpu_stats

MEMORY_STATS = {'cpu' : get_cpu_stats, 'cuda' : get_gpu_stats}

def locate_spins_t0(args) -> Type[Namespace]:
    ith_spin_to_jth_cell = -1 * torch.ones(args.spins_r0.shape[0], 
                                           dtype  = torch.int64,
                                           device = args.spins_r0.device
                                          )
    
    ith_spin_to_jth_fiber = -1 * torch.ones(args.spins_r0.shape[0], 
                                            dtype = torch.int64, 
                                            device = args.spins_r0.device
                                            )
    # True if there are fibers in the voxel. This should be vacuously true, but just to be safe! 
    if any(["fibers_" in k for k in vars(args).keys()]):
 
        f_ctr_dynam = args.fibers_center[None, :, :] + _gamma_t(args.spins_r0, 
                                                                args.fibers_direction, 
                                                                args.fibers_theta, 
                                                                args.fibers_kappa,
                                                                args.fibers_L,
                                                                args.fibers_A,
                                                                args.fibers_P,
                                                                force_broadcast = True
                                                                )   
        f_dir_dynam = _d_gt__d_t(args.spins_r0, 
                                 args.fibers_direction, 
                                 args.fibers_theta, 
                                 args.fibers_kappa,
                                 args.fibers_L,
                                 args.fibers_A,
                                 args.fibers_P, 
                                 force_broadcast = True
                                )

        dist_ith_spin_to_jth_fiber = args.spins_r0[:, None, :] - f_ctr_dynam

        dist_l_2_sqr_ith_spin_to_jth_fiber = torch.square(torch.linalg.norm(dist_ith_spin_to_jth_fiber, axis = -1, ord = 2))

        p     = torch.einsum('NFi, NFi -> NF', 
                            dist_ith_spin_to_jth_fiber, 
                            f_dir_dynam
                            )

        p_sqr = torch.square(p)
        dist_ith_spin_to_jth_fiber_along_p = torch.sqrt(dist_l_2_sqr_ith_spin_to_jth_fiber - p_sqr)

        f_argwhere = torch.argwhere( dist_ith_spin_to_jth_fiber_along_p < args.fibers_radius )  
        ith_spin_to_jth_fiber[f_argwhere[:, 0]] = f_argwhere[:, -1]

    # True if there are cells in the voxel. This should be vacuously true, but just to be safe! 
    if any(["cells_" in k for k in vars(args).keys()]):

        d_ith_spin_to_jth_cell         = args.spins_r0[:, None, :] - args.cells_center[None, :, :]
        d_l_2_sqr_ith_spin_to_jth_cell = torch.linalg.norm(d_ith_spin_to_jth_cell, axis = -1, ord = 2)
        
        c_argwhere = torch.argwhere( d_l_2_sqr_ith_spin_to_jth_cell < args.cells_radius)
        ith_spin_to_jth_cell[c_argwhere[:, 0]] = c_argwhere[:, -1]

    # If a spin is in a fiber and a cell, allocate the spin to the fiber
    ith_spin_to_jth_cell[torch.logical_and(ith_spin_to_jth_fiber > -1, ith_spin_to_jth_cell > -1)] = -1

    water_key = -1 * torch.ones(args.spins_r0.shape[0],
                                dtype = torch.int64,
                                device = args.spins_r0.device
                                )
    
    # If the spin is not in a fiber or in a cell, then it's in the water
    water_key[torch.logical_and(ith_spin_to_jth_fiber < 0, ith_spin_to_jth_cell < 0)] = 1

    # Display MC integrated volume fractions. 
    # Note that because spins may only be in a single element, the MC integrated fractions may not correspond
    # with the configuration file's inputs. This is particularly true for the cells.
    logger.info('-------------------------------')
    logger.info('MC Integration Volume Fractions')
    logger.info('Fiber Volume : {:05f}'.format( (ith_spin_to_jth_fiber > -1).sum() / args.spins_r0.shape[0] ))
    logger.info('Cell  Volume : {:05f}'.format( (ith_spin_to_jth_cell > -1 ).sum() / args.spins_r0.shape[0] ))
    logger.info('Water Volume : {:05f}'.format( (water_key > -1).sum()             / args.spins_r0.shape[0] ))
    logger.info('-------------[QC]--------------')
    logger.info('Unaccounted Spins : {:05d}'.format(args.spins_r0.shape[0] - ((ith_spin_to_jth_fiber > -1).sum() + (ith_spin_to_jth_cell > -1 ).sum() +  (water_key > -1).sum()    ) ) )
    logger.info('-------------------------------')
    
    # write the data to the output Namespace object
    for k,v in zip(KEY_MAP_NAMES, [ith_spin_to_jth_fiber, ith_spin_to_jth_cell, water_key]):
        setattr(args, k, v)
    return args
    
def merge_object_data_and_args(geometry_obj, gradient_obj, args) -> Namespace:
    # Organize and move data to the appropriate device
    inputs_dict = {**geometry_obj.__dict__, **gradient_obj.__dict__, **vars(args)}
    
    # k,v storage for data derived from command-line inputs, and the geometry and gradient objects 
    merged_outputs = {}

    for data_item in RANDOM_WAlK_DATA:
        merged_outputs[data_item] = inputs_dict[data_item]

    return Namespace(merged_outputs)

def collect_and_distribute_data(args):
    # k,v storage for torch.Tensors derrived from the [fiber,cell,spin] objects
    torch_data = {}

    args = vars(args)

    # iterate over [fiber, cell, spin] object
    for obj in OBJECT_TO_TORCH:
        # check if the object is in the voxel, i.e., could have voxel with no cells or no fibers.
        if len(args[obj]) > 0:
            # for each attribute of fiber, cell, and spin, package the data into a numpy array.
            for k, v in args[obj][0].__dict__.items():
                # for each object in the list of objects at args[obj], write the data for that object's attribute to an array, and then convert to a 
                # torch.Tensor for use in the diffusion step
                torch_data[f"{obj}_{k}"] = torch.from_numpy(
                    np.stack([obj.__dict__[k] for obj in args[obj]])
                )
        args.pop(obj)

    # Package remaining numeric simulation data as torch.Tensor types 
    for k, v in args.items():
        if type(v) in {np.ndarray, np.float64, np.float32}:
            torch_data[k] = torch.from_numpy(np.stack([v])).float().squeeze() #Do this to cast np.float64 and np.float32 types as torch.Tensor
            
    # Send torch.Tensor data to the gpu, if indicated in the input command
    if args['use_cuda']:
        for k, v in torch_data.items():
            torch_data[k] = v.to('cuda')
   
    # Send the remaining string / logical data to the torch_data dictionary
    for k in (set(args.keys()) - set(torch_data.keys())):
        torch_data[k] = args[k]
    
    # Finally, configure the directory for hardware usage tracking
    torch_data['monitoring_dir'] = os.path.join(args['sim_out_dir'], 'monitoring')

    if not os.path.exists(torch_data['monitoring_dir']):
        os.mkdir(torch_data['monitoring_dir'])

    return Namespace(torch_data)

def package_data(args) -> List[Type[Namespace]]:
    split_sizes = get_gpu_split_size(args) if args.use_cuda else get_cpu_split_size(args)
    outs = [None] * split_sizes.shape[0]
    
    current = 0
    for j_id, split_size in enumerate(split_sizes):
        start, stop = current, current + split_size
        # k,v storage for the j_id-th batch's diffusion step input arguments
        kth_group_args = {}
        # Group the j_id-th job's spins
        kth_group_args[f"r_0_k"] = args.spins_r0[start:stop]
        # The j_id-th job's spins resident fiber indicies
        kth_group_args[f"r_0_k_spin_in_jth_fiber"] = args.ith_spin_in_jth_fiber_key[start:stop]        
        # The j_id-th job's cells resident cell indicies
        kth_group_args[f"r_0_k_spin_in_jth_cell"] = args.ith_spin_in_jth_cell_key[start:stop]
        # The j_id-th job's water key
        kth_group_args[f"r_0_k_water_key"] = args.water_key[start:stop] > -1
        # The fibers in the j_id-th spin group
        r_0_kth_fibers = (kth_group_args[f"r_0_k_spin_in_jth_fiber"])[kth_group_args[ f"r_0_k_spin_in_jth_fiber"] > -1]
        # The cells in the j_id-th spin group
        r_0_kth_cells = (kth_group_args[f"r_0_k_spin_in_jth_cell"])[kth_group_args[f"r_0_k_spin_in_jth_cell"] > -1]
        
        kth_group_args[f"r_0_k_spin_in_jth_fiber"] = kth_group_args[f"r_0_k_spin_in_jth_fiber"] > -1
        kth_group_args[f"r_0_k_spin_in_jth_cell"]  =  kth_group_args[f"r_0_k_spin_in_jth_cell"] > -1

        # derived fiber parameters
        for k,v in {k:v for k,v in vars(args).items() if "fibers_" in k}.items():
            kth_group_args[f"{k}_k"] = v[r_0_kth_fibers]

        # derived cell parameters
        for k,v in {k:v for k,v in vars(args).items() if "cells_" in k}.items():
            kth_group_args[f"{k}_k"] = v[r_0_kth_cells]

        # dispatch remaining simulation arguments to the kth_group_args
        for k,v in {k:v for k,v in vars(args).items() if not any(["spin" in k, "water" in k])}.items():          
            kth_group_args[k] = v

        # tell working it's job id and how many jobs are in the que
        kth_group_args['job_id'] = j_id + 1
        kth_group_args['n_jobs'] = split_sizes.shape[0]
        # append the j_id-th job's arguments to the outputs
        outs[j_id] = Namespace(kth_group_args)
    return outs

def cleanup_output_data(total_args : Type[Namespace], phases : List[torch.Tensor]) -> Type[Namespace]:
    # merge phases with simulation data : List[torch.Tensor] into a single torch.Tensor 
    total_phases = torch.concatenate(phases, dim = 0)
    setattr(total_args, 'ensemble_phase', total_phases) 
    return total_args

def run_diffusion_process(args) -> torch.Tensor:
    # instantiate the diffusion process object
    dp = DiffusionProcess(args)
    # run the random walk and return the phase
    phase = dp.run()    
    return phase

class DiffusionProcess:
    def __init__(self, args : Type[Namespace]) -> None:
        self.__dict__.update(vars(args))
        self.configure_progress_tracking()
        pass

    def configure_progress_tracking(self) -> None:
        self.csv_path    = os.path.join(self.monitoring_dir, f"mp_job_{self.job_id}.csv")
        self.pid         = os.getpid()
        self.init_time   = time.time()
        self.device_name = torch.cuda.get_device_name() if self.r_0_k.is_cuda else platform.processor()
        self.device_type = 'cuda' if self.r_0_k.is_cuda else 'cpu'
        return
    

    def run(self) -> Dict[str, Union[str, torch.FloatTensor]]:      
        phase = torch.zeros(
            (self.r_0_k.shape[0], self.G.shape[0]),
            dtype=self.r_0_k.dtype,
            device= (self.r_0_k.device).type
        )
        for t in range(self.G.shape[1]):
            if t % 10 == 0:
                self.update_progress_table(t)
            self.step()    
            phase += (
                    GAMMA 
                    * self.dt
                    * torch.einsum('bi, Ni -> Nb', self.G[:, t, :], self.r_0_k)
            )

        self.close_progress_table()
        return phase
    
    def step(self) -> None:
        
        fiber_step(
            self.r_0_k_spin_in_jth_fiber,
            self.r_0_k,
            self.fibers_step_k,
            self.fibers_radius_k,
            self.fibers_center_k,
            self.fibers_direction_k,
            self.fibers_kappa_k,
            self.fibers_L_k,
            self.fibers_A_k,
            self.fibers_P_k,
            self.fibers_theta_k
        )
        
        cell_step(
            self.r_0_k_spin_in_jth_cell,
            self.r_0_k,
            self.fibers_radius,
            self.fibers_center, 
            self.fibers_direction,
            self.fibers_kappa,
            self.fibers_L,
            self.fibers_A,
            self.fibers_P,
            self.fibers_theta,
            self.cells_center_k,
            self.cells_radius_k,
            self.D0
        )
        
        water_step(
            self.r_0_k_water_key,
            self.r_0_k,
            self.fibers_radius,
            self.fibers_center, 
            self.fibers_direction,
            self.fibers_kappa,
            self.fibers_L,
            self.fibers_A,
            self.fibers_P,
            self.fibers_theta,
            self.cells_center,
            self.cells_radius,
            self.D0
        )
        return
    
    def update_progress_table(self, t) -> None:
        curr_time = time.time()
        with open(self.csv_path, 'a', newline = '') as csvfile:                
            
            hardware_query = {
                        'name'               : self.device_name, 
                        'job_id'             : self.job_id,
                        'n_jobs'             : self.n_jobs,
                        'process_id'         : self.pid,
                        'memory.used'        : "{:.4f}".format( float(MEMORY_STATS[self.device_type]()[-2])),
                        'iter'               : t,
                        'total_iter'         : self.G.shape[1] + 1,
                        'time'               : "{:.4f}".format((1e3*(t+1)*self.dt).item()),
                        'total_time'         : "{:.4f}".format((1e3*self.G.shape[1]*self.dt).item()),
                        'world_time'         : "{:.4f}".format((curr_time - self.init_time) / 60)
                        }    
            stats = [str(v) for (k,v) in hardware_query.items()]
            csvwriter = csv.writer(csvfile, csv.QUOTE_ALL)
            csvwriter.writerow(stats)
        return
    
    def close_progress_table(self) -> None:
        curr_time = time.time()
        with open(self.csv_path, 'a', newline = '') as csvfile:      
            cpu_query = {'message'      : r'Done',
                        'elapsed_time' : np.round((curr_time - self.init_time) / 60, 3)
                        }     
            cpu_stats = [str(v) for (k,v) in cpu_query.items()]
            csvwriter = csv.writer(csvfile, csv.QUOTE_ALL)
            csvwriter.writerow(cpu_stats)
        return
    
def compute_random_walk(geometry_obj, gradient_obj, args) -> Type[Namespace]:
    # Parse the input data to get only the arguments required to perform the random walk
    random_walk_args = merge_object_data_and_args(geometry_obj, gradient_obj, args)

    # Package the data and send to the indicated device
    device_args = collect_and_distribute_data(random_walk_args)

    # Locate the initial spin positions
    device_args = locate_spins_t0(device_args)

    # Split the device args n_workers ways. n_workers = 1 for the gpu,
    # and is n_cores if cpu multiprocessing is specified.
    split_device_args = package_data(device_args)

    #launch subprocess to monitor the completion of the random walk from within each process launched by Parallel()
    
    sp = subprocess.Popen(
                        [sys.executable, 
                        f"{os.path.join(os.path.dirname(os.path.realpath(__file__)),'monitor.py')}", 
                        "-csv_dir", 
                        f"{split_device_args[0].monitoring_dir}",
                        "-n_row",
                        f"{len(split_device_args)}"
                        ], 
                        )
    start = time.time()
    # execute the diffusion process
    phases = Parallel(n_jobs = 1 if args.use_cuda else (args.n_cores or (psutil.cpu_count() - 2)), 
                      max_nbytes=None, 
                      mmap_mode=None)(delayed(run_diffusion_process)(arg) for arg in split_device_args)    
    # kill the subprocess. It should terminate anyways by this point, but just to be safe.
    sp.kill()
    end = time.time()
    sys.stdout.write('\n')
    logger.info(f'Random Walk Complete! [Elapsed in {round(end - start, 4)} (sec.)]')
    # cleanup data for save function
    output_args = cleanup_output_data(device_args, phases)
    return output_args