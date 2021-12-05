# The computational domain is [-L/2, L/2]
# solve dq/dt + M(q, dq, ...) = (q_jet - q)/t
# boundary conditions depend on the modeling term (q = q_jet at top/bottom)





# solve dw/dt + M(w) = (w_jet - w)/t 
# boundary conditions depend on the modeling term 
# At top/bottom q is populated as q_jet or q_in depending on b(y)
import scipy.io
import scipy.ndimage
import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Utility import gradient_first, gradient_second, gradient_first_c2f, gradient_first_f2c, interpolate_c2f, interpolate_f2c, psi_fft_sol, gradient


def hyperdiffusion(q, nu, hyper_n, dy):
    q1 = q[0, :]
    q2 = q[1, :]
    
    dnq1 = (-1)**hyper_n * nu*gradient(q1, dy, 2*hyper_n)
    dnq2 = (-1)**hyper_n * nu*gradient(q2, dy, 2*hyper_n)
    
    return -np.vstack((dnq1, dnq2))
    
# the model is a function: w,t ->  M(w)
def explicit_solve(model, q0, f, params, dt = 1.0, Nt = 1000, save_every = 1):
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
    
    for i in range(1, Nt+1): 
        psi = psi_fft_sol(q, F1, F2, dy)
        dd_psi2 = gradient(psi[1, :], dy, 2)
        
        # print(dt, q, model(q, yy, params),  hyperdiffusion(q, nu, hyper_n, dy), dt*mu*dd_psi2)
        
        tend[:,:] = f + model(q, psi, yy, params) + hyperdiffusion(q, nu, hyper_n, dy)
        tend[1,:] -= mu*dd_psi2
        
#         print(tend)
        q += dt * tend
#         q += dt*(f - model(q, yy, params) + hyperdiffusion(q, nu, hyper_n, dy))
#         q[1, :] -= dt*mu*dd_psi2
        
        
        
        if i%save_every == 0:
            q_data[i//save_every, :, :] = q
            t_data[i//save_every] = i*dt
            print(i, "max q", np.max(q))

    return  yy, t_data, q_data


def load_netcdf(pre_file, file_name, start, end, step):
    
    f = pre_file + file_name + '.' + str(start) + '.nc'
    ds = xr.open_dataset(f, engine='h5netcdf')
    _, nlayer, nx, ny = ds.data_vars['q'].values.shape
    
    nt = (end - start)//step + 1
    u, v, q, psi = np.zeros((nt, nx, ny, nlayer)), np.zeros((nt, nx, ny, nlayer)), np.zeros((nt, nx, ny, nlayer)), np.zeros((nt, nx, ny, nlayer))
    for i in range(nt):
        f = pre_file + file_name + '.' + str(start + i*step) + '.nc'
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


# # 7
# def postprocess_mu_helper(u, v, q, psi, L, beta1, beta2):
#     nt, nx, ny, nlayers = u.shape

#     q_zonal_mean = np.mean(q, axis = 1)
#     dq_zonal_mean = np.copy(q_zonal_mean)

#     yy, dy = np.linspace(0, L - L/ny, ny), L/ny

#     flux_zonal_mean = np.mean(v * q, axis = 1)
#     for i in range(nt):
#         for j in range(nlayers):
#             dq_zonal_mean[i, :, j] = gradient_first(q_zonal_mean[i, :, j], dy)

    
#     dpv_zonal_mean = np.copy(dq_zonal_mean)
#     dpv_zonal_mean[:, :, 0] =  dq_zonal_mean[:, :, 0] + beta1
#     dpv_zonal_mean[:, :, 1] =  dq_zonal_mean[:, :, 1] + beta2

#     flux_mean    = np.mean(flux_zonal_mean, axis = 0)
#     dpv_mean     = np.mean(dpv_zonal_mean, axis = 0)
    
#     # todo  further average
#     flux_mean[:, 0] = np.mean(flux_mean[:, 0])
#     mu_mean = flux_mean / dpv_mean

#     return mu_mean.T

# # 4/5 args
# def postprocess_mu_gf(pre_file, L, beta1, beta2, last_n_outputs = 100):
    
#     u = np.load(pre_file + "u_data.npy")
#     v = np.load(pre_file + "v_data.npy")
#     q = np.load(pre_file + "q_data.npy")
#     psi = np.load(pre_file + "psi_data.npy")
    
#     return postprocess_mu_helper(u[last_n_outputs:-1, :, :], v[last_n_outputs:-1, :, :], 
#                           q[last_n_outputs:-1, :, :], psi[last_n_outputs:-1, :, :], 
#                           L, beta1, beta2, last_n_outputs)


# # 8 args
# def postprocess_mu_pyqg(pre_file, file_name, start, end, step, L, beta1, beta2):
    
#     u, v, q, psi = load_netcdf(pre_file, file_name, start, end, step)
    
#     return postprocess_mu_helper(u, v, 
#                           q, psi, 
#                           L, beta1, beta2)


def preprocess_data(file_name, beta, lam, dU, L, start=3000000, end=6000000, step=20000):
    
    F1 = 2.0/lam**2
    F2 = 2.0/lam**2
    beta1, beta2 = beta + F1*dU, beta - F2*dU


    pre_file  = '/central/groups/esm/zhaoyi/pyqg_run/2layer/' + file_name + '/'
    start, end, step = 3000000, 6000000, 20000
    u, v, q, psi = load_netcdf(pre_file, file_name, start, end, step)
    
    nt, nx, ny, nlayers = u.shape

    q_zonal_mean   = np.mean(q, axis = 1)
    psi_zonal_mean = np.mean(psi, axis = 1)
    dq_zonal_mean  = np.copy(q_zonal_mean)
    u_zonal_mean   = np.mean(u, axis = 1)
    vor_zonal_mean = np.copy(u_zonal_mean)
    
    yy, dy = np.linspace(0, L - L/ny, ny), L/ny

    flux_zonal_mean = np.mean(v * q, axis = 1)
    for i in range(nt):
        for j in range(nlayers):
            dq_zonal_mean[i, :, j] = gradient_first(q_zonal_mean[i, :, j], dy)
            vor_zonal_mean[i, :, j] = gradient_first(u_zonal_mean[i, :, j], dy)

    dpv_zonal_mean =  np.copy(dq_zonal_mean)
    dpv_zonal_mean[:,:, 0] += beta1
    dpv_zonal_mean[:,:, 1] += beta2    
    
    t_mean_steps = range(0,nt)
    flux_mean    = np.mean(flux_zonal_mean[t_mean_steps, :, :], axis = 0)
    flux_mean[:, 0] = np.mean(flux_mean[:, 0])
    dpv_mean     = np.mean(dpv_zonal_mean[t_mean_steps, :, :],  axis = 0)
    q_mean       = np.mean(q_zonal_mean[t_mean_steps, :, :],    axis = 0)
    u_mean         = np.mean(u_zonal_mean[t_mean_steps, :, :],    axis = 0)
    vor_mean         = np.mean(vor_zonal_mean[t_mean_steps, :, :],    axis = 0)
    psi_mean = np.mean(psi_zonal_mean[t_mean_steps, :, :],    axis = 0)

    # mu is positive
    mu_mean = -flux_mean / dpv_mean

    
    return [mu_mean.T, dpv_mean.T, u_mean.T, vor_mean.T, q_mean.T, psi_mean.T, flux_mean.T], [q_zonal_mean, dq_zonal_mean]




def nummodel(q, psi, yy, params, mu_c):
    beta, dU, F1, F2 = params["beta"], params["dU"], params["F1"], params["F2"]
    beta1, beta2 = beta + F1*dU, beta - F2*dU
    
    # beta1, beta2 = beta + F1*dU, beta + F2*dU
    
    dy = yy[1] - yy[0]
    
    q1, q2 = q[0, :], q[1, :]
    dq1, dq2 = gradient_first_f2c(q1, dy), gradient_first_f2c(q2, dy)
    
    
    # todo 
    mu_t = mu_c 
    
    
    # mu_t[1,:] = scipy.ndimage.gaussian_filter1d(mu_t[1,:], 5)
    
    J1 = gradient_first_c2f(mu_t[0,:] * (dq1 + beta1), dy)
    J2 = gradient_first_c2f(mu_t[1,:] * (dq2 + beta2), dy)
    
    return np.vstack((J1, J2))


def nummodel_fft(q, psi, yy, params, mu_c):
    beta, dU, F1, F2 = params["beta"], params["dU"], params["F1"], params["F2"]
    beta1, beta2 = beta + F1*dU, beta - F2*dU
    
    # beta1, beta2 = beta + F1*dU, beta + F2*dU
    
    dy = yy[1] - yy[0]
    
    q1, q2 = q[0, :], q[1, :]
    dq1, dq2 = gradient(q1, dy, 1), gradient_first_f2c(q2, dy, 1)
    
    
    # todo 
    mu_t = mu_c 
    
    
    J1 = gradient(mu_t[0,:] * (dq1 + beta1), dy, 1)
    J2 = gradient(mu_t[1,:] * (dq2 + beta2), dy, 1)
    
    return np.vstack((J1, J2))


def nnmodel(q, psi, yy, params, model_top, top_x_normalizer, top_y_normalizer, model_bot, bot_x_normalizer, bot_y_normalizer):
    beta, dU, F1, F2 = params["beta"], params["dU"], params["F1"], params["F2"]
    beta1, beta2 = beta + F1*dU, beta - F2*dU
    dy = yy[1] - yy[0]
    ny = len(yy)
    
    
    q1, q2 = q[0, :], q[1, :]
    dq1, dq2 = gradient_first_f2c(q1, dy), gradient_first_f2c(q2, dy)
    
    dpv = np.copy(q)
    

    top_x = np.zeros((ny*1, 1))
    bot_x = np.zeros((ny*1, 1))
    
    top_x[:, 0] = dq1 + beta1
    bot_x[:, 0] = dq2 + beta2
    
    
    top_x = torch.from_numpy(top_x.astype(np.float32))
    bot_x = torch.from_numpy(bot_x.astype(np.float32))
    
    
    mu_c = np.copy(q)
    mu_c[0, :] = top_y_normalizer.decode(model_top(top_x_normalizer.encode(top_x))).detach().numpy().flatten()
    mu_c[1, :] = bot_y_normalizer.decode(model_bot(bot_x_normalizer.encode(bot_x))).detach().numpy().flatten()
    
    mu_c[mu_c < 0] = 0.0
#     print(mu_c.min())
#     print("mu_c : ", mu_c)
    
    mu_c[0,:] = scipy.ndimage.gaussian_filter1d(mu_c[0,:], 5)
    mu_c[1,:] = scipy.ndimage.gaussian_filter1d(mu_c[1,:], 5)
    
    J1 = gradient_first_c2f(mu_c[0,:] * (dq1 + beta1), dy)
    J2 = gradient_first_c2f(mu_c[1,:] * (dq2 + beta2), dy)

    return np.vstack((J1, J2))

    

def solve_q(Ny, L, F1, F2, beta, mu, dU, hyper_nu, hyper_order, q0,
            dt, Nt, save_every,
            MODEL = "nummodel", mu_mean = [], clip_val = np.inf):
 
    params = {
        "L":    L,
        "dU":   dU,
        "beta": beta,
        "mu":   mu,
        "F1":   F1,
        "F2":   F2,
        "nu":   hyper_nu,
        "hyperdiffusion_order": hyper_order
        }

    beta1, beta2 = beta + F1*dU, beta - F2*dU
    
    # beta1, beta2 = beta + F1*dU, beta + F2*dU
    
    f = np.zeros((2, Ny))
    yy = np.linspace(0, L - L/Ny, Ny)
    
    # initial condition
#     q0 = np.zeros((2, Ny))
#     q0[0, :] = 1e-2 * np.sin(2*np.pi*yy/L)
#     q0[1, :] = 1e-2 * np.cos(2*np.pi*yy/L)

    if MODEL == "nummodel":
        mu_c = np.zeros((2, Ny))
        mu_c[0, :] = interpolate_f2c(mu_mean[0, :])
        mu_c[1, :] = interpolate_f2c(mu_mean[1, :])
        mu_c[mu_c > clip_val] = clip_val
        model = lambda q, psi, yy, params : nummodel(q, psi, yy, params, mu_c)
    elif MODEL == "nnmodel":
        mymodel_top, x_normalizer_top, y_normalizer_top          = torch.load("top_layer.model"),    torch.load("top_layer.model.x_normalizer"),    torch.load("top_layer.model.y_normalizer")
        mymodel_bottom, x_normalizer_bottom, y_normalizer_bottom = torch.load("bottom_layer.model"), torch.load("bottom_layer.model.x_normalizer"), torch.load("bottom_layer.model.y_normalizer")
        model = lambda q, psi, yy, params : nnmodel(q, psi, yy, params, mymodel_top, x_normalizer_top, y_normalizer_top, 
                                                mymodel_bottom, x_normalizer_bottom, y_normalizer_bottom)
    else:
        print("ERROR")


    yy, t_data, q_data = explicit_solve(model, q0, f, params, dt = dt, Nt = Nt, save_every = save_every)
    
    return yy, t_data, q_data
