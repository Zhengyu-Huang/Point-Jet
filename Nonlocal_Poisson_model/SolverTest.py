import scipy.io
import scipy.ndimage
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
import matplotlib.pyplot as plt
from Solver import nummodel, explicit_solve, permeability_ref

L = 1.0
dbc = np.array([0.0, 0.0]) 

def func(xx, A = 10):
    return A * xx


plt.figure()
Ny = 100

for i in [0,1]:
        
    yy, dy = np.linspace(0, L, Ny), L/(Ny - 1)
    delta = 0.2
    if i == 0:
        permeability = lambda x: permeability_ref(x, True, delta/dy)
    else:
        permeability = lambda x: permeability_ref(x, False)
        
    model = lambda q, yy, res : nummodel(permeability, q, yy, res)
    f = func(yy)   
    yy, t_data, q_data = explicit_solve(model, f, dbc, dt = 2.0e-6, Nt = 500000, save_every = 1, L = L)
    
    plt.plot(yy, q_data[-1, :],  "--o", fillstyle="none", label="q-"+str(Ny))


plt.xlabel("y")
plt.legend()
    
plt.show()
 
