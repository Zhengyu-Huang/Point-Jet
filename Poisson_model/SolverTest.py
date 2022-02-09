import scipy.io
import scipy.ndimage
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Solver import nummodel, nummodel_jac, explicit_solve, implicit_solve


Ny = 200
L = 1.0
yy, dy = np.linspace(0, L, Ny), L/(Ny - 1)

TEST = "D=dtheta**2-case1"
if TEST == "Linear":
    def permeability(q, dq):
        return 0.5  + np.zeros(dq.size)   
    f = np.sin(2*np.pi*yy)
    q_sol = 1/0.5 * 1/(2*np.pi)**2 * f
    dbc = np.array([0.0, 0.0])
    
elif TEST == "D=dtheta**2-case1":
    def permeability(q, dq):
        return dq**2  
    f = 6*(-2*yy + 1)**2
    q_sol = -yy*(yy - 1)
    dbc = np.array([0.0, 0.0])
    
elif TEST == "D=dtheta**2-case2":
    def permeability(q, dq):
        return dq**2  
    f = 3*(np.pi*np.sin(np.pi * yy))**2 * (np.pi*np.cos(np.pi * yy))
    q_sol = np.cos(np.pi * yy)
    dbc = np.array([1.0, -1.0])
       
elif TEST == "D=dtheta**2+1-case1":
    # works
    def permeability(q, dq):
        return dq**2 + 1
    f = 6*(-2*yy + 1)**2 + 2.0
    q_sol = -yy*(yy - 1)
    dbc = np.array([0.0, 0.0])
          
elif TEST == "D=theta**2-case1":
    def permeability(q, dq):
        return q**2 + 1
    f = -2*(np.pi**2)*np.cos(np.pi*yy)*(np.sin(np.pi*yy)**2) + (np.pi**2)*(np.cos(np.pi*yy)**3)
    q_sol = np.cos(np.pi * yy)
    dbc = np.array([1.0, -1.0]) 
  
elif TEST == "D=theta**2-case3":
    def permeability(q, dq):
        return q**2
    f = np.zeros(yy.shape)
    q_sol = yy**(1/3)
    dbc = np.array([0.0, 1.0])     
    




yy = np.linspace(0, L, Ny)
MODEL = "imp_nummodel"


if MODEL == "exp_nummodel":

    model = lambda q, yy, res : nummodel(permeability, q, yy, res)
    yy, t_data, q_data = explicit_solve(model, f, dbc, dt = 1.0e-4, Nt = 10000, save_every = 1, L = L)

elif MODEL == "imp_nummodel":
    
    model = lambda q, yy, dt, res, V : nummodel_jac(permeability, q, yy, dt, res, V)  
    # yy, t_data, q_data = implicit_solve(model, f, dbc, dt = 1.0e-3, Nt = 10000, save_every = 1, L = L)
    yy, t_data, q_data = implicit_solve(model, f, dbc, dt = 1.0e-2, Nt = 1, save_every = 1, L = L)
       
else:
    print("ERROR")


plt.figure()
# plt.plot(yy, np.mean(q_data[0, :], axis=0),  label="top")
# plt.plot(yy, np.mean(q_data[1, :], axis=0),  label="bottom")

# plt.plot(yy, q_data[-1, :],  "--o", label="q")
# plt.plot(yy, q_sol,  "--*", label="q ref")


plt.plot(yy, q_data[-1, :],  "--o", fillstyle="none", label="q")
plt.plot(yy, q_sol,  "--*", label="q ref")
# plt.plot(yy, f,  "--o", fillstyle="none", label="f")

plt.xlabel("y")
plt.legend()
plt.show()
 
