import pandas as pd
import numpy as np
import os 



lines = []

bvals = pd.read_csv(r"C:\Users\Administrator\Desktop\NODDI_bval.csv", header=None).to_numpy()
bvecs = pd.read_csv(r"C:\Users\Administrator\Desktop\NODDI_bvec.csv", header=None).to_numpy()
shells = np.unique(bvals)
lines.append('[shells = {}]'.format(shells.shape[0]))
lines.append('')
for i, shell in enumerate(shells):
    lines.append('[bvalue = {}]'.format(shell))
    bvecsAtShell = bvecs[np.where(bvals == shell)[0],:]
    lines.append('[directions = {}]'.format(bvecsAtShell.shape[0]))
    lines.append('[CordinateSystem = xyz]')
    lines.append('Normalisation = unity')
    for j in range(bvecsAtShell.shape[0]):
        lines.append('Vector[{}] = ( {},{},{} )'.format(j, bvecsAtShell[j,0], bvecsAtShell[j,1], bvecsAtShell[j,2]))

with open(r"C:\Users\Administrator\Desktop\NODDI_145_b4000.txt", 'w') as f:
    f.write('\n'.join(lines))


