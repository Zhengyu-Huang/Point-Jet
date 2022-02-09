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
def explicit_solve(model, f, dbc, dt = 1.0, Nt = 1000, save_every = 1, L = 1.0):
    
    Ny = f.size
    yy = np.linspace(0, L, Ny)
    
    t = 0.0
    # qi are periodic (qi[0] == qi[-1])
    q = np.zeros(Ny)
    q[0], q[-1] = dbc[0], dbc[1]
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
def implicit_solve(model_jac, f, dbc, dt = 1.0, Nt = 1000, save_every = 1, L = 1.0):
    
    Ny = f.size
    yy = np.linspace(0, L, Ny)
    
    t = 0.0
    # qi are periodic (qi[0] == qi[-1])
    q = np.zeros(Ny)
    # q = -yy*(yy - 1)
    
    q[0], q[-1] = dbc[0], dbc[1]
    
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
            print(i, "max q: ", np.max(q), " L2 res: ", np.linalg.norm(f[1:Ny-1] + res))

    return  yy, t_data, q_data


 
def nummodel(permeability, q, yy, res):
    
    Ny = yy.size
    dy = yy[1] - yy[0]
    dq_c = gradient_first_f2c(q, dy)
    q_c = interpolate_f2c(q)
    mu_c = permeability(q_c, dq_c)
    res[:] = gradient_first_c2f(mu_c*(dq_c), dy)
    

def nummodel_jac(permeability, q, yy, dt,  res, V):
    
    Ny = yy.size
    dy = yy[1] - yy[0]
    dq_c = gradient_first_f2c(q, dy)
    q_c = interpolate_f2c(q) 
    mu_c = permeability(q_c, dq_c)
    res[:] = gradient_first_c2f(mu_c*(dq_c), dy)
    
    V[:] = 0
    
    for i in range(Ny-2):
        if i == 0:
            V[0], V[1] =  1 + dt/dy**2 * (mu_c[i] +  mu_c[i+1]), -dt/dy**2 * mu_c[i+1]
        elif i == Ny-3:
            V[2 + 3*(i - 1) + 0], V[2 + 3*(i - 1) + 1] = -dt/dy**2 * mu_c[i], 1 + dt/dy**2 * (mu_c[i] +  mu_c[i+1])
        else:
            V[2 + 3*(i - 1) + 0], V[2 + 3*(i - 1) + 1], V[2 + 3*(i - 1) + 2] = -dt/dy**2 * mu_c[i], 1 + dt/dy**2 * (mu_c[i] +  mu_c[i+1]), -dt/dy**2 * mu_c[i+1]
      
    return res, V


# def nnmodel_jac(torchmodel, omega, tau, dy):
#     # return np.zeros(len(omega))
#     # a0, a1, b0, b1 = 0.2, 0.3, 0.4, 0.5

#     d_omega = gradient_first_c2f(omega, dy)

#     omega_f = interpolate_c2f(omega)
#     input  = torch.from_numpy(np.stack((abs(omega_f), d_omega)).T.astype(np.float32))
#     mu_f = -torchmodel(input).detach().numpy().flatten()
#     mu_f[mu_f >= 0.0] = 0.0

#     mu_f = scipy.ndimage.gaussian_filter1d(mu_f, 5)
    
#     M = gradient_first_f2c(mu_f*(d_omega), dy)

#     return M



