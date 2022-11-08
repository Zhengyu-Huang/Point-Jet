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
from Numerics import gradient_first_c2f, gradient_first_f2c, interpolate_f2c, gradient_first
import NeuralNet



#########################################
# Neural network information
#########################################
ind, outd, width = 2, 1, 10
layers = 2
activation, initializer, outputlayer = "sigmoid", "default", "None"

non_negative = True
filter_on    = True
filter_sigma = 5.0

mu_scale     = 1.0
flux_scale   = 1.0
source_scale = 1.0


def permeability_ref(x):
    q, dq = x[:, 0], x[:, 1]
    return np.sqrt(q**2 + dq**2) 
def D_permeability_ref(x):
    q, dq = x[:, 0], x[:, 1]
    return q/np.sqrt(q**2 + dq**2), dq/np.sqrt(q**2 + dq**2)

def flux_ref(x):
    q,  dq = x[:, 0], x[:, 1]
    return np.sqrt(q**2 + dq**2)*dq 

def source_ref(x):
    q,  dq, ddq = x[:, 0], x[:, 1], x[:, 2]
    return np.sqrt(q**2 + dq**2)*ddq + (q*dq+dq*ddq)*dq/np.sqrt(q**2 + dq**2)

def source_ref_q(q, dy):
    q_c,  dq_c = interpolate_f2c(q), gradient_first_f2c(q, dy)
    flux_c = np.sqrt(q_c**2 + dq_c**2)*dq_c
    source = np.copy(q)
    source[1:-1] = gradient_first_f2c(flux_c, dy)
    source[0] = source[1]
    source[-1] = source[-2]
    return source

    
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
#         model(q, yy, res)
#         q[1:Ny-1] += dt*(f[1:Ny-1] + res)
        
        
        q_old = np.copy(q)
        model(q, yy, res)
        q[1:Ny-1] += dt*(f[1:Ny-1] + res)
        
        model(q, yy, res)
        q[1:Ny-1] = q_old[1:Ny-1]/2 + q[1:Ny-1]/2  + dt/2*(f[1:Ny-1] + res)
        
        
        
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
    q = np.linspace(dbc[0], dbc[1], Ny)
    # q = -yy*(yy - 1)
    # q[0], q[-1] = dbc[0], dbc[1]
    
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
        model_jac(q, yy, res, V)
        # print("error : ", np.linalg.norm(sparse.coo_matrix((V,(I,J)),shape=(Ny-2,Ny-2)).tocsc() * q[1:Ny-1] - res) )
        
        V *= -dt
        V[0::3] += 1.0
        
        A = sparse.coo_matrix((V,(I,J)),shape=(Ny-2,Ny-2)).tocsc()
        dq = spsolve(A, dt*(f[1:Ny-1] + res))
        q[1:Ny-1] += dq

        
        if i%save_every == 0:
            q_data[i//save_every, :] = q
            t_data[i//save_every] = i*dt
            print(i, "max q: ", np.max(q), " L2 res: ", np.linalg.norm(f[1:Ny-1] + res))
        if i == Nt:
            print("error dq = ", np.linalg.norm(dq/dt))
            
    return  yy, t_data, q_data


# the model is a function: q,t ->  M(q)
# the solution points are at cell faces
def implicit_Newton_solve(model_jac, f, dbc, dt = 1.0, Nt = 1000, save_every = 1, L = 1.0, Newton_eps = 1e-3, Newton_maxiterstep = 1000):
    
    
    
    Ny = f.size
    yy = np.linspace(0, L, Ny)
    
    t = 0.0
    # qi are periodic (qi[0] == qi[-1])
    q = np.linspace(dbc[0], dbc[1], Ny)
    # q = -yy*(yy - 1)
    # q[0], q[-1] = dbc[0], dbc[1]
    
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
        
        Newton_iterstep, Newton_converge = 0, False
        
        q_old = np.copy(q)
        
        while  not Newton_converge:

            Newton_iterstep += 1
            
            model_jac(q, yy, res, V)
            
            V *= -dt
            V[0::3] += 1.0
            A = sparse.coo_matrix((V,(I,J)),shape=(Ny-2,Ny-2)).tocsc()

            res_all = q[1:Ny-1] - q_old[1:Ny-1]  - dt*(f[1:Ny-1] + res)
            
            delta_q = spsolve(A, res_all)

            q[1:Ny-1] -= delta_q
            # @show Newton_iterstep, norm(res)
            if (np.linalg.norm(res_all) < Newton_eps  or Newton_iterstep > Newton_maxiterstep):
                if Newton_iterstep > Newton_maxiterstep:
                    print("Newton iteration does not converge :", Newton_iterstep, " error is : ", np.linalg.norm(res_all))
                Newton_converge = True
            


        
        if i%save_every == 0:
            q_data[i//save_every, :] = q
            t_data[i//save_every] = i*dt
            print(i, "max q: ", np.max(q), " L2 res: ", np.linalg.norm(res_all))

    return  yy, t_data, q_data



 
def nummodel(permeability, q, yy, res):
    
    Ny = yy.size
    dy = yy[1] - yy[0]
    dq_c = gradient_first_f2c(q, dy)
    q_c = interpolate_f2c(q)
    x = np.vstack((q_c, dq_c)).T
    mu_c = permeability(x = x)
    res[:] = gradient_first_c2f(mu_c*(dq_c), dy)

# dM/dx    
def nummodel_flux(flux, q, yy, res):
    
    Ny = yy.size
    dy = yy[1] - yy[0]
    dq_c = gradient_first_f2c(q, dy)
    q_c = interpolate_f2c(q)
    x = np.vstack((q_c, dq_c)).T
    M_c = flux(x = x)
    res[:] = gradient_first_c2f(M_c, dy)

    
def nummodel_source(source, q, yy, res):
    
    Ny = yy.size
    dy = yy[1] - yy[0]
    dq = gradient_first(q, dy)
    ddq = gradient_first(dq, dy)
    x = np.vstack((q, dq, ddq)).T
    S_f = source(x = x)
    res[:] = S_f[1:-1]
    
def nummodel_jac(permeability, q, yy, res, V, exact = False, D_permeability = None):
    
    Ny = yy.size
    dy = yy[1] - yy[0]
    dq_c = gradient_first_f2c(q, dy)
    q_c = interpolate_f2c(q)
    x = np.vstack((q_c, dq_c)).T
    mu_c = permeability(x = x)
    
    # i -> i-1/2
    #   q: 0 ---- 1 ---- 2 ---- 3 ---- 4   ...   ---- Ny-2 ---- Ny-1
    # q_c:     0      1      2      3      ...             Ny-2
    # jac:        0      1      2      3   ...        Ny-3   
    
    
    res[:] = gradient_first_c2f(mu_c*(dq_c), dy)
    
    V[:] = 0
    
   
    for i in range(Ny-2):
        if i == 0:
            V[0], V[1] =  -1/dy**2 * (mu_c[i] +  mu_c[i+1]), 1/dy**2 * mu_c[i+1]
        elif i == Ny-3:
            V[2 + 3*(i - 1) + 0], V[2 + 3*(i - 1) + 1] = 1/dy**2 * mu_c[i], -1/dy**2 * (mu_c[i] +  mu_c[i+1])
        else:
            V[2 + 3*(i - 1) + 0], V[2 + 3*(i - 1) + 1], V[2 + 3*(i - 1) + 2] = 1/dy**2 * mu_c[i], -1/dy**2 * (mu_c[i] +  mu_c[i+1]), 1/dy**2 * mu_c[i+1]
    if exact:
        Dq_c, Ddq_c = D_permeability(q_c, dq_c)
        
        for i in range(Ny-2):
            # diagonal i,i : q_{i+1}
            V[3*i] += (q[i+2] - q[i+1])/dy**2 * (1/2 * Dq_c[i+1] - Ddq_c[i+1]/dy) - (q[i+1] - q[i])/dy**2 * (1/2 * Dq_c[i] + Ddq_c[i]/dy)
        for i in range(Ny-3):
            # +1 diagonal
            V[3*i + 1] += (q[i+2] - q[i+1])/dy**2 * (1/2 * Dq_c[i+1] + Ddq_c[i+1]/dy)
            # -1 diagonal
            V[3*(i + 1) - 1] -= (q[i+2] - q[i+1])/dy**2 * (1/2 * Dq_c[i+1] - Ddq_c[i+1]/dy)
        
               
    return res, V





def generate_data_helper(permeability, f_func, L=1.0, Nx = 100):
    xx = np.linspace(0.0, L, Nx)
    dy = xx[1] - xx[0]
    f = f_func(xx)   
    dbc = np.array([0.0, 0.0]) 
       
    model = lambda q, yy, res : nummodel(permeability, q, yy, res)
    model_jac = lambda q, yy, res, V : nummodel_jac(permeability, q, yy, res, V, exact = False, D_permeability = None)
    
    
    #xx, t_data, q_data = explicit_solve(model, f, dbc, dt = 5.0e-5, Nt = 5000, save_every = 1000, L = L)
    xx, t_data, q_data = implicit_solve(model_jac, f, dbc, dt = 5.0e-3, Nt = 2000, save_every = 1000, L = L)

    
#     print("Last step increment is : ", np.linalg.norm(q_data[-1, :] - q_data[-2, :]), " last step is : ", np.linalg.norm(q_data[-1, :]))
    
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