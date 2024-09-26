from progress_table import ProgressTable
import glob as glob 
import os 
import numpy as np
import argparse
import csv

parser = argparse.ArgumentParser(description='You can add a description here')
parser.add_argument('-csv_dir', '--dir', help='path_to_csv_folder', required=True)
parser.add_argument('-n_rows', '--n_rows', help = 'number of rows', required=True)
args = vars(parser.parse_args())
table = ProgressTable(
    pbar_show_progress=False,
    pbar_show_throughput=False,
    pbar_show_eta=False,
    pbar_show_percents=True,
    pbar_embedded=False,
    pbar_style="angled alt red blue",
    default_column_width=15,
    default_header_color="bold",
)

table.add_columns("Process", "Job", "Iteration", "t [Msec.]", "Memory [Mb]", "Elapsed Time [Min.]")
table.add_rows(2 * int(args['n_rows']), color="red")


pbars = [table.pbar(1,
                    description = "Job {}".format(idx + 1), 
                    position = 2 * idx + 1, 
                    static=True, 
                    ) 
        for idx in range(int(args['n_rows']))]

cond  = np.zeros(int(args['n_rows']), dtype='bool')
while (~cond.all()):
    files = glob.glob(os.path.join(args['dir'], '*.csv'))
    for index, file in enumerate(files):
        with open(file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
            if len(rows) > 0:
                current_data = rows[-1]
                if current_data[0] != 'Done':
                    job_index = int(current_data[1]) -1 
                    
                    table.update("Job", 
                                f"{current_data[1]} / {current_data[2]}", 
                                row = 2 * job_index
                                )
                    
                    table.update('Process', 
                                f"{current_data[3]}", 
                                row = 2 * job_index
                                )
                    
                    table.update('Iteration',
                                f"{current_data[5]} / {current_data[6]}", 
                                row = 2 * job_index
                                )
                    
                    table.update('t [Msec.]', 
                                f"{ current_data[-3]}",
                                row = 2 * job_index
                                )
                    
                    table.update('Memory [Mb]', 
                            f"{current_data[4]}",
                            row = 2 * job_index
                            )
                    
                    table.update('Elapsed Time [Min.]', 
                                f"{current_data[-1]}", 
                                row = 2 * job_index
                                )
                    
                    pbars[job_index].reset( float(current_data[5]) / float(current_data[6]) )
    
                else:
                    cond[index] = True

