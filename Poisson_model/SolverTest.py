import scipy.io
import scipy.ndimage
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Solver import nummodel, nummodel_jac, explicit_solve, implicit_solve


Ny = 100
L = 1.0
yy, dy = np.linspace(0, L, Ny), L/(Ny - 1)

TEST = "D=theta**2"
if TEST == "Linear":
    def permeability(q):
        return 0.5  + np.zeros(q.size)   
    f = np.sin(2*np.pi*yy)
    q_sol = 1/0.5 * 1/(2*np.pi)**2 * f
elif TEST == "D=theta**2":
    def permeability(q):
        return q**2  
    f = (10*yy**4 - 20*yy**3 + 12*yy**2 - 2*yy)
    q_sol = -yy*(yy - 1)



yy = np.linspace(0, L, Ny)
MODEL = "imp_nummodel"


if MODEL == "exp_nummodel":

    model = lambda q, yy, res : nummodel(permeability, q, yy, res)
    yy, t_data, q_data = explicit_solve(model, f, dt = 1.0e-4, Nt = 10000, save_every = 1, L = L)

elif MODEL == "imp_nummodel":
    
    model = lambda q, yy, dt, res, V : nummodel_jac(permeability, q, yy, dt, res, V)  
    yy, t_data, q_data = implicit_solve(model, f, dt = 1.0e-2, Nt = 10000, save_every = 1, L = L)
       
else:
    print("ERROR")


plt.figure()
# plt.plot(yy, np.mean(q_data[0, :], axis=0),  label="top")
# plt.plot(yy, np.mean(q_data[1, :], axis=0),  label="bottom")

plt.plot(yy, q_data[-1, :],  "--o", label="q")
plt.plot(yy, q_sol,  "--*", label="q ref")

plt.xlabel("y")
plt.legend()
plt.show()
 
