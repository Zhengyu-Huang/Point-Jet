# The computational domain is [0, 1]
# solve d/dt(q) + d/dx( D(q) d/dx(q) ) = f(x)
# boundary conditions are periodic


import scipy.io
import scipy.ndimage
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Utility import gradient_first_c2f, gradient_first_f2c, interpolate_f2c
from NeuralNet import *


    
    
# the model is a function: q,t ->  M(q)
# the solution points are at cell faces
def explicit_solve(model, f,  dt = 1.0, Nt = 1000, save_every = 1, L = 1.0):
    
    Ny = f.size
    yy = np.linspace(0, L, Ny)
    
    t = 0.0
    # qi are periodic (qi[0] == qi[-1])
    q = np.zeros(Ny)
    
    # prepare data storage
    q_data = np.zeros((Nt//save_every+1, Ny))
    t_data = np.zeros(Nt//save_every+1)
    q_data[0, :], t_data[0] = q, t

    res = np.zeros(Ny - 2)

    for i in range(1, Nt+1): 
        model(q, yy, res)
        q[1:Ny-1] += dt*(f[1:Ny-1] + res)
        
        if i%save_every == 0:
            q_data[i//save_every, :] = q
            t_data[i//save_every] = i*dt
            print(i, "max q", np.max(q))

    return  yy, t_data, q_data


# the model is a function: q,t ->  M(q)
# the solution points are at cell faces
def implicit_solve(model_jac, f,  dt = 1.0, Nt = 1000, save_every = 1, L = 1.0):
    
    Ny = f.size
    yy = np.linspace(0, L, Ny)
    
    t = 0.0
    # qi are periodic (qi[0] == qi[-1])
    q = np.zeros(Ny)
    
    # prepare data storage
    q_data = np.zeros((Nt//save_every+1, Ny))
    t_data = np.zeros(Nt//save_every+1)
    q_data[0, :], t_data[0] = q, t

    I, J = np.zeros(3*(Ny-2)-2), np.zeros(3*(Ny-2)-2)
    for i in range(Ny-2):
        if i == 0:
            I[0], I[1] = i, i 
            J[0], J[1] = i, i+1
        elif i == Ny-3:
            I[2 + 3*(i - 1) + 0], I[2 + 3*(i - 1) + 1] = i, i
            J[2 + 3*(i - 1) + 0], J[2 + 3*(i - 1) + 1] = i-1, i
        else:
            I[2 + 3*(i - 1) + 0], I[2 + 3*(i - 1) + 1], I[2 + 3*(i - 1) + 2] = i, i, i
            J[2 + 3*(i - 1) + 0], J[2 + 3*(i - 1) + 1], J[2 + 3*(i - 1) + 2] = i-1, i, i+1
            
            
    res = np.zeros(Ny-2)  
    V = np.zeros(3*(Ny-2)-2)
        
    for i in range(1, Nt+1): 
        
        # this include dt in both V
        model_jac(q, yy, dt, res, V)
        A = sparse.coo_matrix((V,(I,J)),shape=(Ny-2,Ny-2)).tocsc()
        q[1:Ny-1] += spsolve(A, dt*(f[1:Ny-1] + res))
        
        
        if i%save_every == 0:
            q_data[i//save_every, :] = q
            t_data[i//save_every] = i*dt
            print(i, "max q", np.max(q))

    return  yy, t_data, q_data


 
def nummodel(q, yy, res):
    
    Ny = yy.size
    dy = yy[1] - yy[0]
    dq_c = gradient_first_f2c(q, dy)
    q_c = interpolate_f2c(q)
    mu_c = permeability(q_c)
    res[:] = gradient_first_c2f(mu_c*(dq_c), dy)
    

def nummodel_jac(q, yy, dt, res, V):
    
    Ny = yy.size
    dy = yy[1] - yy[0]
    dq = gradient_first_f2c(q, dy)
    q_c = interpolate_f2c(q)
    mu_c = permeability(q_c)
    res[:] = gradient_first_c2f(mu_c*(dq), dy)
    
    V[:] = 0
    
    for i in range(Ny-2):
        if i == 0:
            V[0], V[1] =  1 + dt/dy**2 * (mu_c[i] +  mu_c[i+1]), -dt/dy**2 * mu_c[i+1]
        elif i == Ny-3:
            V[2 + 3*(i - 1) + 0], V[2 + 3*(i - 1) + 1] = -dt/dy**2 * mu_c[i], 1 + dt/dy**2 * (mu_c[i] +  mu_c[i+1])
        else:
            V[2 + 3*(i - 1) + 0], V[2 + 3*(i - 1) + 1], V[2 + 3*(i - 1) + 2] = -dt/dy**2 * mu_c[i], 1 + dt/dy**2 * (mu_c[i] +  mu_c[i+1]), -dt/dy**2 * mu_c[i+1]
      
    return res, V


def nnmodel(torchmodel, omega, tau, dy):
    # return np.zeros(len(omega))
    # a0, a1, b0, b1 = 0.2, 0.3, 0.4, 0.5

    d_omega = gradient_first_c2f(omega, dy)

    omega_f = interpolate_c2f(omega)
    input  = torch.from_numpy(np.stack((abs(omega_f), d_omega)).T.astype(np.float32))
    mu_f = -torchmodel(input).detach().numpy().flatten()
    mu_f[mu_f >= 0.0] = 0.0

    mu_f = scipy.ndimage.gaussian_filter1d(mu_f, 5)
    
    M = gradient_first_f2c(mu_f*(d_omega), dy)

    return M

def permeability(q):
    # return q * (1 - q)
    return 0.5  + np.zeros(q.size)   

Ny = 100
L = 1.0
yy, dy = np.linspace(0, L, Ny), L/(Ny - 1)
f = np.sin(2*np.pi*yy)
yy = np.linspace(0, L, Ny)
MODEL = "imp_nummodel"

q_sol = 1/0.5 * 1/(2*np.pi)**2 * f

if MODEL == "exp_nummodel":

    model = lambda q, yy, res : nummodel(q, yy, res)
    yy, t_data, q_data = explicit_solve(model, f, dt = 1.0e-4, Nt = 10000, save_every = 1, L = L)

elif MODEL == "imp_nummodel":
    
    model = lambda q, yy, dt, res, V : nummodel_jac(q, yy, dt, res, V)  
    yy, t_data, q_data = implicit_solve(model, f, dt = 1.0e-2, Nt = 100, save_every = 1, L = L)
      
elif MODEL == "nnmodel":
    
    mymodel = torch.load("visc.model")
    model = lambda omega, tau, dy : nnmodel(mymodel, omega, tau, dy)
    
else:
    print("ERROR")


plt.figure()
# plt.plot(yy, np.mean(q_data[0, :], axis=0),  label="top")
# plt.plot(yy, np.mean(q_data[1, :], axis=0),  label="bottom")

plt.plot(yy, q_data[-1, :],  label="q")
plt.plot(yy, q_sol,  label="q ref")

plt.xlabel("y")
plt.legend()
plt.show()
 
