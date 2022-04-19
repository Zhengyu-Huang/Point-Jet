import scipy.io
import scipy.ndimage
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
import matplotlib.pyplot as plt
from Solver import nummodel, explicit_solve

L = 1.0
dbc = np.array([0.0, 0.0]) 

def func(xx, A = 10):
    return A * xx


def permeability_ref(x, delta_dy):
    q, dq = x[:, 0], x[:, 1]
    mu = np.sqrt(q**2 + dq**2) 

    mu =  scipy.ndimage.gaussian_filter1d(mu, delta_dy, mode="nearest") 
            
    return mu


plt.figure()
Ny = 100

for i in [0,1]:
    if i == 0:
        def permeability_ref(x, delta_dy):
            q, dq = x[:, 0], x[:, 1]
            mu = np.sqrt(q**2 + dq**2) 

            mu =  scipy.ndimage.gaussian_filter1d(mu, delta_dy, mode="nearest") 
                    
            return mu
    else:
        def permeability_ref(x, delta_dy):
            q, dq = x[:, 0], x[:, 1]
            mu = np.sqrt(q**2 + dq**2)         
            return mu

        
    yy, dy = np.linspace(0, L, Ny), L/(Ny - 1)
    delta = 0.2
    model = lambda q, yy, res : nummodel(permeability_ref, q, yy, res, delta)
    f = func(yy)   
    yy, t_data, q_data = explicit_solve(model, f, dbc, dt = 2.0e-6, Nt = 500000, save_every = 1, L = L)
    
    plt.plot(yy, q_data[-1, :],  "--o", fillstyle="none", label="q-"+str(Ny))


plt.xlabel("y")
plt.legend()
    
plt.show()
 
