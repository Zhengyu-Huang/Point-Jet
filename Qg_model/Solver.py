import scipy.io
import scipy.ndimage
import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
sys.path.append('../Utility')
from Numerics import gradient_first,  gradient_first_c2f, gradient_first_f2c, interpolate_c2f, interpolate_f2c, psi_fft_sol, gradient_fft, precomp_fft
import NeuralNet







#########################################
# Neural network information
#########################################
ind, outd, width = 3, 1, 10
layers = 2
activation, initializer, outputlayer = "sigmoid", "default", "tanh"
mu_scale = 100.0
flux_scale = 100.0
non_negative = True
filter_on = True
filter_sigma = 5.0
# input scale
q_scale = 100
dpv_scale = 10
psi_scale = 100
u_scale = 100
mu_low = 0.1

def str_to_num(x):
    if x.find("p") == -1:
        return np.float64(x)
    
    int_part, frac_part = x.split("p")
    return np.float64(int_part) + np.float64(frac_part)/10.0**(len(frac_part))

def load_netcdf(folder_name, file_name, start, end, step):
    
    f = folder_name + file_name + '.' + str(start) + '.nc'
    ds = xr.open_dataset(f, engine='h5netcdf')
    _, nlayer, nx, ny = ds.data_vars['q'].values.shape
    
    nt = (end - start)//step + 1
    u, v, q, psi = np.zeros((nt, nx, ny, nlayer)), np.zeros((nt, nx, ny, nlayer)), np.zeros((nt, nx, ny, nlayer)), np.zeros((nt, nx, ny, nlayer))
    for i in range(nt):
        f = folder_name + file_name + '.' + str(start + i*step) + '.nc'
        ds = xr.open_dataset(f, engine='h5netcdf')
        psi_h_i = ds.data_vars['ph']
        psi_i = np.fft.irfftn(psi_h_i, axes=(-2,-1))
        
        u[i, :, :, 0]   = ds.data_vars['u'].values[0, 0, :, :].T
        u[i, :, :, 1]   = ds.data_vars['u'].values[0, 1, :, :].T
        v[i, :, :, 0]   = ds.data_vars['v'].values[0, 0, :, :].T
        v[i, :, :, 1]   = ds.data_vars['v'].values[0, 1, :, :].T
        q[i, :, :, 0]   = ds.data_vars['q'].values[0, 0, :, :].T
        q[i, :, :, 1]   = ds.data_vars['q'].values[0, 1, :, :].T
        psi[i, :, :, 0] = psi_i[0, 0, :, :].T
        psi[i, :, :, 1] = psi_i[0, 1, :, :].T
    
    
    return u, v, q, psi




def preprocess_data(folder_name, file_name, beta, dU, L, start=3000000, end=6000000, step=20000):
    
    beta1 = beta2 = beta

    u, v, q, psi = load_netcdf(folder_name, file_name, start, end, step)
    
    nt, nx, ny, nlayers = u.shape

    q_zonal_mean   = np.mean(q, axis = 1)
    psi_zonal_mean = np.mean(psi, axis = 1)
    dq_zonal_mean  = np.copy(q_zonal_mean)
    u_zonal_mean   = np.mean(u, axis = 1)
    vor_zonal_mean = np.copy(u_zonal_mean)
    
    yy, dy = np.linspace(L/(2*ny), L - L/(2*ny), ny), L/ny

    flux_zonal_mean = -np.mean(v * q, axis = 1)
    for i in range(nt):
        for j in range(nlayers):
            dq_zonal_mean[i, :, j] = gradient_first(q_zonal_mean[i, :, j], dy, bc="periodic")
            vor_zonal_mean[i, :, j] = gradient_first(u_zonal_mean[i, :, j], dy, bc="periodic")
    
    
    # compute psi variance
    psi_var_2 = np.copy(psi)
    for i in range(nt):
        for j in range(nx):
            psi_var_2[i, j, :, :] -= psi_zonal_mean[i, :, :]
            
    psi_var_2 **= 2
    psi_var_2_zonal_mean = np.mean(psi_var_2, axis = 1)
    
    
    
    dpv_zonal_mean =  np.copy(dq_zonal_mean)
    dpv_zonal_mean[:,:, 0] += beta1
    dpv_zonal_mean[:,:, 1] += beta2    
    
    t_mean_steps = range(0,nt)
    flux_mean    = np.mean(flux_zonal_mean[t_mean_steps, :, :], axis = 0)
    # flux_mean[:, 0] = np.mean(flux_mean[:, 0])
    dpv_mean     = np.mean(dpv_zonal_mean[t_mean_steps, :, :],  axis = 0)
    
    
    
    q_mean       = np.mean(q_zonal_mean[t_mean_steps, :, :],    axis = 0)
    u_mean         = np.mean(u_zonal_mean[t_mean_steps, :, :],    axis = 0)
    vor_mean         = np.mean(vor_zonal_mean[t_mean_steps, :, :],    axis = 0)
    psi_mean = np.mean(psi_zonal_mean[t_mean_steps, :, :],    axis = 0)
    psi_var_2_mean = np.mean(psi_var_2_zonal_mean[t_mean_steps, :, :],    axis = 0)
    mu_mean = flux_mean / dpv_mean

    
    return [mu_mean.T, dpv_mean.T, u_mean.T, vor_mean.T, q_mean.T, psi_mean.T, flux_mean.T, psi_var_2_mean.T], [q_zonal_mean, dq_zonal_mean]






class QG_params:
    def __init__(self, L, dU, F1, F2, hyper_order, hyper_nu, 
                 beta, rek):
        self.L = L
        self.dU = dU
        self.F1 = F1
        self.F2 = F2
        self.hyper_order= hyper_order
        self.hyper_nu = hyper_nu
        
        self.beta = beta
        self.rek = rek
    

def load_data(beta_rek_strs, beta_reks):

    Ny = 256
    L = 50*2*np.pi
    H = [1.0, 1.0]               # the rest depths of each layer
    kd = 1.0                     # rd

    U = [0.0, 0.0]
    dU = U[0] - U[1] 
    F1 = kd/(1 + (H[0]/H[1])**2.0)
    F2 = kd/(1 + (H[1]/H[0])**2.0)
    hyper_nu, hyper_order = 0.0, 2
    Q = 1.0

    yy, dy = np.linspace(L/(2*Ny), L - L/(2*Ny), Ny), L/Ny

    force = np.zeros((2, Ny))
    force[0, :] = -Q * np.sin(2*np.pi*yy/L)
    force[1, :] =  Q * np.sin(2*np.pi*yy/L)
    
    N_runs = len(beta_reks)
    
    # todo rek  = 0.X, beta = X
    # betas = [str(beta_reks[i][0]) for i in range(N_runs)]
    # reks = ["rek0p" + str(np.int64(beta_reks[i][1]*10)) for i in range(N_runs)]
    
    file_names = ["nx256beta" + beta_rek_strs[i][0] + "rek" + beta_rek_strs[i][1]  for i in range(N_runs)] 
    folder_names = ["/central/groups/esm/zhaoyi/pyqg_run/2layer/inhomogeneous/" + file_names[i] + "/" for i in range(N_runs)]
    

    start, end, step = 500000, 1000000, 20000

    N_data = len(folder_names)
    mu_mean,  closure_mean,  dpv_mean, q_mean, psi_mean, u_mean = np.zeros((N_data, 2, Ny)), np.zeros((N_data, 2, Ny)), np.zeros((N_data, 2, Ny)), np.zeros((N_data, 2, Ny)), np.zeros((N_data, 2, Ny)), np.zeros((N_data, 2, Ny))


    for i in range(N_data):  

        flow_means, flow_zonal_means = preprocess_data(folder_names[i], file_names[i], beta_reks[i][0], dU, L, start, end, step)
        mu_mean[i, :, :], dpv_mean[i, :, :], u_mean[i, :, :], vor_mean, q_mean[i, :, :], psi_mean[i, :, :], closure_mean[i, :, :], psi_var_2_mean = flow_means[:8]
    
    
    mu_mean_clip = np.copy(mu_mean)
    # TODO: clean data
    mu_mean_clip[mu_mean_clip <= 0.0 ] = 0.0
    for i in range(N_data):
        for layer in range(2):
            mu_mean_clip[i, layer, :] = scipy.ndimage.gaussian_filter1d(mu_mean_clip[i, layer, :], 5)

    
    physics_params = [QG_params(L=L, dU=dU, F1=F1, F2=F2, hyper_nu=hyper_nu, hyper_order=hyper_order, beta=beta, rek=rek)
     for beta, rek in beta_reks]
    
    return physics_params, q_mean, psi_mean, dpv_mean,  mu_mean, mu_mean_clip,  closure_mean, yy, force







################################################################################################################################
def hyperdiffusion(q, nu, hyper_n, dy):
    q1 = q[0, :]
    q2 = q[1, :]
    
    dnq1 = (-1)**hyper_n * nu*gradient_fft(q1, dy, 2*hyper_n)
    dnq2 = (-1)**hyper_n * nu*gradient_fft(q2, dy, 2*hyper_n)
    
    return -np.vstack((dnq1, dnq2))


# def nummodel(permeability, beta1, beta2, q, psi, yy, res):
    
#     dy = yy[1] - yy[0]
    
#     # from face to center and then back to face
#     q1, q2 = interpolate_f2c(q[0, :], bc="periodic"), interpolate_f2c(q[1, :], bc="periodic")
#     dq1, dq2 = gradient_first_f2c(q1, dy, bc="periodic"), gradient_first_f2c(q2, dy, bc="periodic")
    
#     q = np.hstack((q1,q2)) / q_scale
# #     dq = np.hstack((dq1,dq2)) / dpv_scale
# #     x = np.vstack((q , dq)).T
#     dpv = np.hstack((dq1 + beta1,dq2 + beta2)) / dpv_scale
#     x = np.vstack((q , dpv)).T
    
#     mu = permeability(x = x)
#     mu_c1 = mu[0: len(yy)]
#     mu_c2 = mu[len(yy):]
    
#     res[0, :] = gradient_first_c2f(mu_c1 * (dq1 + beta1), dy, bc="periodic")
#     res[1, :] = gradient_first_c2f(mu_c2 * (dq2 + beta2), dy, bc="periodic")
    



def nummodel_fft(permeability, beta1, beta2, q, psi, yy, res, k2, dealiasing_filter):
    
    dy = yy[1] - yy[0]
    
    # all are at the cell center
    
    dq1, dq2 = gradient_fft(q[0, :], dy, 1, k2, dealiasing_filter), gradient_fft(q[1, :], dy, 1, k2, dealiasing_filter)
    dpv = np.hstack((dq1 + beta1,dq2 + beta2))   
    
    x = np.vstack((np.fabs(q).flatten()/q_scale, dpv/dpv_scale, np.fabs(psi).flatten()/psi_scale)).T
    
    # x = np.vstack((dpv, np.fabs(psi).flatten()/psi_scale)).T
    
    # x = dpv.reshape((-1, 1))
    
    mu = permeability(x = x)  + mu_low 
    mu_c1 = mu[0: len(yy)]
    mu_c2 = mu[len(yy):]
    
    res[0, :] = gradient_fft(mu_c1 * (dq1 + beta1), dy, 1, k2, dealiasing_filter)
    res[1, :] = gradient_fft(mu_c2 * (dq2 + beta2), dy, 1, k2, dealiasing_filter)

    
    
def nummodel_flux_fft(flux_model, beta1, beta2, q, psi, yy, res, k2, dealiasing_filter):
    
    dy = yy[1] - yy[0]
    
    # all are at the cell center
    
    dq1, dq2 = gradient_fft(q[0, :], dy, 1, k2, dealiasing_filter), gradient_fft(q[1, :], dy, 1, k2, dealiasing_filter)
    dpv = np.hstack((dq1 + beta1, dq2 + beta2))   
    
    x = np.vstack((np.fabs(q).flatten()/q_scale, dpv/dpv_scale, np.fabs(psi).flatten()/psi_scale)).T
    
    
    flux = flux_model(x = x)
    flux_c1 = flux[0: len(yy)]
    flux_c2 = flux[len(yy):]
    
    res[0, :] = gradient_fft(flux_c1, dy, 1, k2, dealiasing_filter)
    res[1, :] = gradient_fft(flux_c2, dy, 1, k2, dealiasing_filter)
    
 

    


def explicit_solve(model_fft, f, q0, params, dt = 1.0, Nt = 1000, save_every = 1):
    L, dU, F1, F2, beta, rek = params.L, params.dU, params.F1, params.F2, params.beta, params.rek
    hyper_nu, hyper_order = params.hyper_nu, params.hyper_order
    
    _, Ny = q0.shape
    yy, dy = np.linspace(L/(2*Ny), L - L/(2*Ny), Ny), L/Ny
    
    t = 0.0
    
    q = np.copy(q0)
    
    # prepare data storage
    q_data = np.zeros((Nt//save_every+1, 2, Ny))
    t_data = np.zeros(Nt//save_every+1)
    q_data[0, :, :], t_data[0] = q, t
    
    tend = np.zeros((2, Ny))
    res = np.zeros((2, Ny))
    
    k2, dealiasing_filter = precomp_fft(Ny)
    
    for i in range(1, Nt+1): 
        psi = psi_fft_sol(q, F1, F2, dy, k2, dealiasing_filter)
        dd_psi2 = gradient_fft(psi[1, :], dy, 2, k2, dealiasing_filter)

        model_fft(q, psi, yy, res, k2, dealiasing_filter)
        tend[:,:] = f + res          # Turn off hyperdiffusion    + hyperdiffusion(q, hyper_nu, hyper_order, dy)
        tend[1,:] -= rek*dd_psi2

        q += dt * tend
        
        if i%save_every == 0:
            q_data[i//save_every, :, :] = q
            t_data[i//save_every] = i*dt  
            
        if i == Nt or i == Nt//2:
            print(i, "max q", np.max(q))

    return  yy, t_data, q_data








    


