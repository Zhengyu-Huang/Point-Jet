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
import torch
import sys
sys.path.append('../Utility')
from Numerics import gradient_first_c2f, gradient_first_f2c, interpolate_f2c
import NeuralNet



#########################################
# Neural network information
#########################################
ind, outd, width = 2, 1, 10
layers = 2
activation, initializer, outputlayer = "sigmoid", "default", "None"
mu_scale = 2.0
non_negative = True
filter_on=True
filter_sigma = 5.0




def permeability_ref(x, filter_sigma=5.0):
    q, dq = x[:, 0], x[:, 1]
    mu = np.sqrt(q**2 + dq**2) 
    mu =  scipy.ndimage.gaussian_filter1d(mu, filter_sigma, mode="nearest")         
    return mu


# the model is a function: q,t ->  M(q)
# the solution points are at cell faces
def explicit_solve(model, f, dbc, dt = 1.0, Nt = 1000, save_every = 1, L = 1.0):
    
    Ny = f.size
    yy = np.linspace(0, L, Ny)
    
    t = 0.0
    # q has Dirichlet boundary condition 
    q = np.linspace(dbc[0], dbc[1], Ny)
    # q[0], q[-1] = dbc[0], dbc[1]
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





 
def nummodel(permeability, q, yy, res):
    
    Ny = yy.size
    dy = yy[1] - yy[0]
    dq_c = gradient_first_f2c(q, dy)
    q_c = interpolate_f2c(q)
    x = np.vstack((q_c, dq_c)).T
    
    mu_c = permeability(x = x)

    res[:] = gradient_first_c2f(mu_c*(dq_c), dy)

# # dM/dx    
# def nummodel_flux(flux, q, yy, res):
    
#     Ny = yy.size
#     dy = yy[1] - yy[0]
#     dq_c = gradient_first_f2c(q, dy)
#     q_c = interpolate_f2c(q)
#     x = np.vstack((q_c, dq_c)).T
#     M_c = flux(x = x)
#     res[:] = gradient_first_c2f(M_c, dy)

    


def generate_data_helper(permeability, f_func, L=1.0, Nx = 100):
    xx = np.linspace(0.0, L, Nx)
    dy = xx[1] - xx[0]
    f = f_func(xx)   
    dbc = np.array([0.0, 0.0]) 
       
    model = lambda q, yy, res : nummodel(permeability, q, yy, res)
    xx, t_data, q_data = explicit_solve(model, f, dbc, dt = 5.0e-6, Nt = 500000, save_every = 100000, L = L)

    
    print("Last step increment is : ", np.linalg.norm(q_data[-1, :] - q_data[-2, :]), " last step is : ", np.linalg.norm(q_data[-1, :]))
    
    q = q_data[-1, :]
    q_c, dq_c = interpolate_f2c(q), gradient_first_f2c(q, dy)
    return xx, f, q, q_c, dq_c 

def generate_data():
    f_funcs = []
    n_data = 10

    for i in range(1,n_data):
        def func(xx, A = i):
            return A * xx
        f_funcs.append(func)


    L = 1.0
    Nx = 100
    n_data = len(f_funcs)
    xx, f, q, q_c, dq_c = np.zeros((n_data, Nx)), np.zeros((n_data, Nx)), np.zeros((n_data, Nx)), np.zeros((n_data, Nx-1)), np.zeros((n_data, Nx-1))


    for i in range(n_data):
        xx[i, :], f[i, :], q[i, :], q_c[i, :], dq_c[i, :] = generate_data_helper(permeability_ref, f_funcs[i], L=L, Nx=Nx)

        
    return xx, f, q, q_c, dq_c