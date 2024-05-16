import numpy as np 
import matplotlib.pyplot as plt 


Da  = 3.0 # um^2 / ms 
Dr  = 0.01 # um^2 / ms


Nb = 90

G = np.zeros((90, 3))
G[0:30,  0] = 1.0
G[30:60, 1] = 1.0
G[60:90, 2] = 1.0

D = np.diag([Dr, Da, Dr])
Delta = 10.0 # ms 
delta = .001 # ms 
tau  = Delta - (delta / 3 )
b   = np.linspace(0, 2, Nb)


print(b.shape, G.shape, D.shape)


s = np.exp(np.einsum('s, si, sj, ij -> s', -b, G, G, D))




q   = np.sqrt( b / (4*np.pi**2 * tau)  )


fig, ax = plt.subplots()

ax.plot(b, s, 'rx')


plt.show()