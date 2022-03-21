import scipy
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
import torch
import pickle
from functools import partial

# from NeuralNet import *
from timeit import default_timer

from Solver import *
import sys
sys.path.append('../Utility')
import NeuralNet
import KalmanInversion 
from Numerics import interpolate_f2c, gradient_first_f2c
# import imp
# imp.reload(KalmanInversion )
# imp.reload(NeuralNet )


# # Load Training data


beta_rek_strs = [("1", "0p3"), ("2", "0p3"), ("3", "0p3"), ("1", "0p6"), ("2", "0p6"), ("3", "0p6")]
beta_reks = [ (str_to_num(beta_rek_strs[i][0]), str_to_num(beta_rek_strs[i][1])) for i in range(len(beta_rek_strs)) ]
phy_params, q_mean, psi_mean, dpv_mean,  mu_mean, mu_mean_clip,  closure_mean, xx, force = load_data(beta_rek_strs = beta_rek_strs, beta_reks = beta_reks)


class QGParam:
    def __init__(self, phy_params, xx, dt, Nt, save_every,  N_y):
        self.theta_names = ["hyperparameters"]
        self.xx = xx
        self.force  = force  
        self.phy_params = phy_params
        
        self.dt = dt
        self.Nt = Nt
        self.save_every = save_every
        
        N_theta = 2
        self.N_theta = N_theta
        
        self.N_y = N_y + N_theta 
        
        
def loss_aug(s_param, params):
    
    dt, Nt, save_every = s_param.dt,  s_param.Nt,   s_param.save_every
    xx = s_param.xx
    force = s_param.force

    _,  Nx = force.shape
    N_data = len(s_param.phy_params)
    
    q_sol = np.zeros((N_data, 2, Nx))
    
    
    L = s_param.phy_params[0].L

    
    def mu_model(x):
        mu_c = np.zeros(2*Nx)
        mu_c[0:Nx] = params[0]
        mu_c[Nx:2*Nx] = params[1]
        return mu_c

    
    for i in range(N_data):
        
        q0 = np.zeros((2, Nx))
        q0[0, :] = -s_param.phy_params[i].beta*L/2 * np.sin(2*np.pi*xx/L)
        q0[1, :] =  s_param.phy_params[i].beta*L/2* np.sin(2*np.pi*xx/L)
    

        beta1 = beta2 = s_param.phy_params[i].beta
        model = lambda q, psi, xx, res : nummodel_fft(mu_model, beta1, beta2,  q, psi, xx, res)
        xx, t_data, q_data = explicit_solve(model, force, q0, s_param.phy_params[i], dt = dt, Nt = Nt, save_every = save_every)

        # TODO
        q_sol[i, :, :] = np.mean(q_data[Nt//(2*save_every):, :, :], axis=0)
        
    return np.hstack((q_sol.flatten(), params))








N_data = len(beta_reks)
y = q_mean.flatten()
Sigma_eta = np.fabs(q_mean)
for i in range(N_data):
    Sigma_eta[i, :] = np.mean(Sigma_eta[i, :])
Sigma_eta = np.diag(np.reshape((Sigma_eta*0.01)**2, -1))

L = 50*2*np.pi
N_y = len(y)
Nx = 256
xx, dy = np.linspace(L/(2*Nx), L - L/(2*Nx), Nx), L/Nx
dt = 4e-3 
save_every = 1000
Nt = 400000 


s_param = QGParam(phy_params, xx, dt, Nt, save_every, N_y)
N_theta = s_param.N_theta


theta0_mean_init = np.array([10.0, 20.0])

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

save_folder = "indirect_const"
uki_obj = KalmanInversion.UKI_Run(s_param, loss_aug, 
    theta0_mean, theta0_mean_init, 
    theta0_cov,  theta0_cov_init,
    y_aug, Sigma_eta_aug,
    alpha_reg,
    gamma,
    update_freq, 
    N_iter,
    save_folder = save_folder)




