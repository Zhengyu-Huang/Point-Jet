import scipy.io
import scipy.ndimage
import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
sys.path.append('../Utility')
from Numerics import gradient_first,  gradient_first_c2f, gradient_first_f2c, interpolate_c2f, interpolate_f2c, psi_fft_sol, gradient_fft
import NeuralNet

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



def hyperdiffusion(q, nu, hyper_n, dy):
    q1 = q[0, :]
    q2 = q[1, :]
    
    dnq1 = (-1)**hyper_n * nu*gradient_fft(q1, dy, 2*hyper_n)
    dnq2 = (-1)**hyper_n * nu*gradient_fft(q2, dy, 2*hyper_n)
    
    return -np.vstack((dnq1, dnq2))


def nummodel(permeability, beta1, beta2, q, psi, yy, res):
    
    dy = yy[1] - yy[0]
    q1, q2 = q[0, :], q[1, :]
    dq1, dq2 = gradient_first_f2c(q1, dy, bc="periodic"), gradient_first_f2c(q2, dy, bc="periodic")
    
    mu_c1, mu_c2 = permeability(q1, dq1, q2, dq2)
    
    res[0, :] = gradient_first_c2f(mu_c1 * (dq1 + beta1), dy, bc="periodic")
    res[1, :] = gradient_first_c2f(mu_c2 * (dq2 + beta2), dy, bc="periodic")
    

def nummodel_fft(permeability, beta1, beta2, q, psi, yy, res):
    
    dy = yy[1] - yy[0]
    q1, q2 = q[0, :], q[1, :]
    dq1, dq2 = gradient_fft(q1, dy, 1), gradient_fft(q2, dy, 1)
    
    mu_c1, mu_c2 = permeability(q1, dq1, q2, dq2)
    
    res[0, :] = gradient_fft(mu_c1 * (dq1 + beta1), dy, 1)
    res[1, :] = gradient_fft(mu_c2 * (dq2 + beta2), dy, 1)
    
    



# def nnmodel(q, psi, yy, params, model_top, top_x_normalizer, top_y_normalizer, model_bot, bot_x_normalizer, bot_y_normalizer):
#     beta, dU, F1, F2 = params["beta"], params["dU"], params["F1"], params["F2"]
#     beta1, beta2 = beta + F1*dU, beta - F2*dU
#     dy = yy[1] - yy[0]
#     ny = len(yy)
    
    
#     q1, q2 = q[0, :], q[1, :]
#     dq1, dq2 = gradient_first_f2c(q1, dy, bc="periodic"), gradient_first_f2c(q2, dy, bc="periodic")
    
#     dpv = np.copy(q)
    

#     top_x = np.zeros((ny*1, 1))
#     bot_x = np.zeros((ny*1, 1))
    
#     top_x[:, 0] = dq1 + beta1
#     bot_x[:, 0] = dq2 + beta2
    
    
#     top_x = torch.from_numpy(top_x.astype(np.float32))
#     bot_x = torch.from_numpy(bot_x.astype(np.float32))
    
    
#     mu_c = np.copy(q)
#     mu_c[0, :] = top_y_normalizer.decode(model_top(top_x_normalizer.encode(top_x))).detach().numpy().flatten()
#     mu_c[1, :] = bot_y_normalizer.decode(model_bot(bot_x_normalizer.encode(bot_x))).detach().numpy().flatten()
    
#     mu_c[mu_c < 0] = 0.0
    
#     mu_c[0,:] = scipy.ndimage.gaussian_filter1d(mu_c[0,:], 5)
#     mu_c[1,:] = scipy.ndimage.gaussian_filter1d(mu_c[1,:], 5)
    
#     J1 = gradient_first_c2f(mu_c[0,:] * (dq1 + beta1), dy, bc="periodic")
#     J2 = gradient_first_c2f(mu_c[1,:] * (dq2 + beta2), dy, bc="periodic")

#     return np.vstack((J1, J2))


def explicit_solve(model, f, q0, params, dt = 1.0, Nt = 1000, save_every = 1):
    L, dU, beta, mu, F1, F2 = params["L"], params["dU"], params["beta"], params["mu"], params["F1"], params["F2"]
    nu, hyper_n = params["nu"], params["hyperdiffusion_order"]
    
    _, Ny = q0.shape
    yy, dy = np.linspace(0, L - L/Ny, Ny), L/Ny
    
    t = 0.0
    
    q = np.copy(q0)
    
    # prepare data storage
    q_data = np.zeros((Nt//save_every+1, 2, Ny))
    t_data = np.zeros(Nt//save_every+1)
    q_data[0, :, :], t_data[0] = q, t
    
    tend = np.zeros((2, Ny))
    res = np.zeros((2, Ny))
    
    for i in range(1, Nt+1): 
        psi = psi_fft_sol(q, F1, F2, dy)
        dd_psi2 = gradient_fft(psi[1, :], dy, 2)
        
        model(q, psi, yy, res)
        tend[:,:] = f + res + hyperdiffusion(q, nu, hyper_n, dy)
        tend[1,:] -= mu*dd_psi2
        
        q += dt * tend
        
        if i%save_every == 0:
            q_data[i//save_every, :, :] = q
            t_data[i//save_every] = i*dt
            print(i, "max q", np.max(q))

    return  yy, t_data, q_data


# def solve(Ny, L, F1, F2, beta, mu, dU, hyper_nu, hyper_order, q0,
#             dt, Nt, save_every,
#             MODEL = "nummodel", mu_mean = [], clip_val = np.inf):
 
#     params = {
#         "L":    L,
#         "dU":   dU,
#         "beta": beta,
#         "mu":   mu,
#         "F1":   F1,
#         "F2":   F2,
#         "nu":   hyper_nu,
#         "hyperdiffusion_order": hyper_order
#         }

#     beta1, beta2 = beta + F1*dU, beta - F2*dU
    
#     # beta1, beta2 = beta + F1*dU, beta + F2*dU
    
#     f = np.zeros((2, Ny))
#     yy = np.linspace(0, L - L/Ny, Ny)
    


#     if MODEL == "nummodel":
#         mu_c = np.zeros((2, Ny))
#         mu_c[0, :] = interpolate_f2c(mu_mean[0, :], bc="periodic")
#         mu_c[1, :] = interpolate_f2c(mu_mean[1, :], bc="periodic")
#         mu_c[mu_c > clip_val] = clip_val
#         model = lambda q, psi, yy, params : nummodel(q, psi, yy, params, mu_c)
#     elif MODEL == "nnmodel":
#         mymodel_top, x_normalizer_top, y_normalizer_top          = torch.load("top_layer.model"),    torch.load("top_layer.model.x_normalizer"),    torch.load("top_layer.model.y_normalizer")
#         mymodel_bottom, x_normalizer_bottom, y_normalizer_bottom = torch.load("bottom_layer.model"), torch.load("bottom_layer.model.x_normalizer"), torch.load("bottom_layer.model.y_normalizer")
#         model = lambda q, psi, yy, params : nnmodel(q, psi, yy, params, mymodel_top, x_normalizer_top, y_normalizer_top, 
#                                                 mymodel_bottom, x_normalizer_bottom, y_normalizer_bottom)
#     else:
#         print("ERROR")


#     yy, t_data, q_data = explicit_solve(model, q0, f, params, dt = dt, Nt = Nt, save_every = save_every)
    
#     return yy, t_data, q_data





    


