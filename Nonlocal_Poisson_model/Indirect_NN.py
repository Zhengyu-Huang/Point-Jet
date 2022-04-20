
# coding: utf-8

# In[7]:


import scipy.io
import scipy.ndimage
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.linalg import block_diag

import matplotlib as mpl
mpl.use('Agg')


import matplotlib.pyplot as plt
import matplotlib.animation as animation
from timeit import default_timer
import torch
from functools import partial




from Solver import *


import sys
sys.path.append('../Utility')
import NeuralNet
import KalmanInversion 
from Numerics import interpolate_f2c, gradient_first_f2c
# import imp
# imp.reload(KalmanInversion )
# imp.reload(NeuralNet )



# jupyter nbconvert --to script 'Indirect_NN.ipynb'





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


f_funcs = []
n_data = 10

for i in range(1,n_data):
    def func(xx, A = i):
        return A * xx
    f_funcs.append(func)
    
    
L = 1.0
Nx = 100
dx = L/(Nx - 1)
n_data = len(f_funcs)
xx, f, q, q_c, dq_c = np.zeros((n_data, Nx)), np.zeros((n_data, Nx)), np.zeros((n_data, Nx)), np.zeros((n_data, Nx-1)), np.zeros((n_data, Nx-1))

GENERATE_DATA = True
if GENERATE_DATA:
    delta = 0.2
    premeability = lambda x : permeability_ref(x, delta/dx)
    for i in range(n_data):
        xx[i, :], f[i, :], q[i, :], q_c[i, :], dq_c[i, :] = generate_data_helper(permeability, f_funcs[i], L=L, Nx=Nx)
        
    np.save("xx.npy",   xx)
    np.save("f.npy",    f)
    np.save("q.npy",    q)
    np.save("q_c.npy",  q_c)
    np.save("dq_c.npy", dq_c)
else:
    xx, f, q, q_c, dq_c = np.load("xx.npy"), np.load("f.npy"), np.load("q.npy"), np.load("q_c.npy"), np.load("dq_c.npy")

# visualize data

plt.figure()
for i in range(n_data):
    plt.plot(q_c[i, :], dq_c[i, :],  "--o", fillstyle="none")

plt.xlabel("q")
plt.ylabel("dq")
plt.title("Training Data")
plt.savefig("Poisson-Training-Data.png")


# # Training Loss : || d(D dq/dy)/dy + f(x)|| on the quadratic function

# In[ ]:


def loss_aug(s_param, params):
    
    
    ind, outd, width   = s_param.ind, s_param.outd, s_param.width
    activation, initializer, outputlayer = s_param.activation, s_param.initializer, s_param.outputlayer
    
    dt, Nt, save_every = s_param.dt,  s_param.Nt,   s_param.save_every
    xx, f = s_param.xx, s_param.f
    dbc = s_param.dbc
    
    
    N_data, Nx = f.shape
    q_sol = np.zeros((N_data, Nx))
    
    delta = params[0]
    dx = xx[1] - xx[0]
    net =  NeuralNet.create_net(ind, outd, layers, width, activation, initializer, outputlayer,  params[1:])
    nn_model = partial(NeuralNet.nn_viscosity, net=net, mu_scale=mu_scale, non_negative=non_negative, filter_on=filter_on, filter_sigma=delta/dx)
    model = lambda q, xx, res : nummodel(nn_model, q, xx, res, delta)
    
    for i in range(N_data):
        _, t_data, q_data = explicit_solve(model, f[i, :], dbc, dt = dt, Nt = Nt, save_every = save_every, L = L)
        q_sol[i, :] = q_data[-1, :]
        
    return np.hstack((np.reshape(q_sol, -1), params))


# ## Start UKI

# In[ ]:


class PoissonParam:
    def __init__(self, 
                 xx, f, dbc, dt, Nt, save_every,
                 N_y, ind, outd, layers, width, activation, initializer, outputlayer 
                 ):
        self.theta_names = ["hyperparameters"]
        
        self.ind  = ind
        self.outd = outd
        self.width = width
        self.activation = activation
        self.initializer = initializer
        self.outputlayer = outputlayer
        
        self.dt = dt
        self.Nt = Nt
        self.save_every = save_every

        self.xx = xx
        self.f  = f
        self.dbc = dbc
        
        
        N_theta = ind*width + (layers - 2)*width**2 + width*outd + (layers - 1)*width + outd if layers > 1 else ind*outd + outd
        # the length scale parameter
        self.N_theta = N_theta + 1
        
        
        self.N_y = N_y + N_theta


# In[ ]:


y = np.reshape(q, -1)
Sigma_eta = np.fabs(q)
for i in range(n_data):
    Sigma_eta[i, :] = np.mean(Sigma_eta[i, :])
Sigma_eta = np.diag(np.reshape((Sigma_eta*0.01)**2, -1))

dbc = np.array([0.0, 0.0])
N_y = len(y)
dt, Nt, save_every = 2.0e-6,  500000, 500000
s_param = PoissonParam(xx, f, dbc, dt, Nt, save_every,
                       N_y, ind, outd, layers, width, activation, initializer, outputlayer)


N_theta = s_param.N_theta

theta0_mean_init = NeuralNet.FNN(ind, outd, layers, width, activation, initializer, outputlayer).get_params()
theta0_mean_init = np.insert(theta0_mean_init, 0, 0.1)
theta0_mean = np.zeros(N_theta)

theta0_cov = np.zeros((N_theta, N_theta))
np.fill_diagonal(theta0_cov, 100.0**2)  
theta0_cov_init = np.zeros((N_theta, N_theta))
np.fill_diagonal(theta0_cov_init, 0.1**2)  

y_aug = np.hstack((y, theta0_mean))
Sigma_eta_aug = block_diag(Sigma_eta, theta0_cov)



alpha_reg = 1.0
update_freq = 1
N_iter = 50
gamma = 1.0


uki_obj = KalmanInversion.UKI_Run(s_param, loss_aug, 
    theta0_mean, theta0_mean_init, 
    theta0_cov, theta0_cov_init, 
    y_aug, Sigma_eta_aug,
    alpha_reg,
    gamma,
    update_freq, 
    N_iter,
    save_folder = "indirect_NN")


