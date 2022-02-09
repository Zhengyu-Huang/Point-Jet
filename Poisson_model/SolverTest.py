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

TEST = "D=theta**2+dtheta**2+1-case2"
if TEST == "Linear":
    def permeability(q, dq):
        return 0.5  + np.zeros(dq.size)   
    f = np.sin(2*np.pi*yy)
    q_sol = 1/0.5 * 1/(2*np.pi)**2 * f
    dbc = np.array([0.0, 0.0])
    
elif TEST == "D=dtheta**2-case1":
    # require very small time step: explicit dt = 5.0e-6, Nt = 1000000
    def permeability(q, dq):
        return dq**2
    f = 6*(-2*yy + 1)**2
    q_sol = -yy*(yy - 1)
    dbc = np.array([0.0, 0.0])
    
elif TEST == "D=dtheta**2-case2":
    # require very small time step: explicit dt = 1.0e-7, Nt = 200000
    def permeability(q, dq):
        return dq**2  
    f = 3 * np.pi**4 * np.sin(np.pi * yy)**2 * np.cos(np.pi * yy)
    q_sol = np.cos(np.pi * yy)
    dbc = np.array([1.0, -1.0])
       
elif TEST == "D=dtheta**2+1-case1":
    # works: implicit dt = 1.0e-3, Nt = 1000
    def permeability(q, dq):
        return dq**2 + 1
    f = 6*(-2*yy + 1)**2 + 2.0
    q_sol = -yy*(yy - 1)
    dbc = np.array([0.0, 0.0])
    
elif TEST == "D=dtheta**2+1-case2":
    # require very small time step: explicit dt = 1.0e-7, Nt = 100000
    # require very small time step: implicit dt = 1.0e-6, Nt = 100000
    def permeability(q, dq):
        return dq**2 + 1
    f = 3 * np.pi**4 * np.sin(np.pi * yy)**2 * np.cos(np.pi * yy) + np.cos(np.pi*yy) * np.pi**2
    q_sol = np.cos(np.pi * yy)
    dbc = np.array([1.0, -1.0]) 
        
elif TEST == "D=theta**2+dtheta**2+1-case1":
    # works: implicit dt = 1.0e-3, Nt = 1000
    def permeability(q, dq):
        return q**2 + dq**2 + 1
    f = 6*(1-2*yy)**2 - 2*(yy - yy**2)*(1 - 2*yy)**2 + 2*(yy - yy**2)**2 + 2 
     
    q_sol = -yy*(yy - 1)
    dbc = np.array([0.0, 0.0]) 
  
elif TEST == "D=theta**2+dtheta**2+1-case2":
    # require very small time step: explicit dt = 1.0e-7, Nt = 100000
    # require very small time step: implicit dt = 1.0e-6, Nt = 20000
    def permeability(q, dq):
        return q**2 + dq**2 + 1
    c , s = np.cos(np.pi*yy), np.sin(np.pi*yy)
    f = np.pi**2*c**3 - 2*np.pi**2*s**2*c + 3*np.pi**4*s**2*c + np.pi**2*c
    
    q_sol = np.cos(np.pi * yy)
    dbc = np.array([1.0, -1.0])     
    




yy = np.linspace(0, L, Ny)
MODEL = "imp_nummodel"


if MODEL == "exp_nummodel":

    model = lambda q, yy, res : nummodel(permeability, q, yy, res)
    yy, t_data, q_data = explicit_solve(model, f, dbc, dt = 1.0e-7, Nt = 100000, save_every = 1, L = L)

elif MODEL == "imp_nummodel":
    
    model = lambda q, yy, dt, res, V : nummodel_jac(permeability, q, yy, dt, res, V)  
    yy, t_data, q_data = implicit_solve(model, f, dbc, dt = 1.0e-6, Nt = 20000, save_every = 1, L = L)
       
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
 
