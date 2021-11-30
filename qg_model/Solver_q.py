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
        
        tend[:,:] = f - model(q, yy, params) + hyperdiffusion(q, nu, hyper_n, dy)
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


# 7
def postprocess_mu_helper(u, v, q, psi, L, beta1, beta2):
    nt, nx, ny, nlayers = u.shape

    q_zonal_mean = np.mean(q, axis = 1)
    dq_zonal_mean = np.copy(q_zonal_mean)

    yy, dy = np.linspace(0, L - L/ny, ny), L/ny

    flux_zonal_mean = np.mean(v * q, axis = 1)
    for i in range(nt):
        for j in range(nlayers):
            dq_zonal_mean[i, :, j] = gradient_first(q_zonal_mean[i, :, j], dy)

    
    dpv_zonal_mean = np.copy(dq_zonal_mean)
    dpv_zonal_mean[:, :, 0] =  dq_zonal_mean[:, :, 0] + beta1
    dpv_zonal_mean[:, :, 1] =  dq_zonal_mean[:, :, 1] + beta2

    flux_mean    = np.mean(flux_zonal_mean, axis = 0)
    dpv_mean     = np.mean(dpv_zonal_mean, axis = 0)
    
    # todo  further average
    flux_mean[:, 0] = np.mean(flux_mean[:, 0])
    mu_mean = flux_mean / dpv_mean

    return mu_mean.T

# 4/5 args
def postprocess_mu_gf(pre_file, L, beta1, beta2, last_n_outputs = 100):
    
    u = np.load(pre_file + "u_data.npy")
    v = np.load(pre_file + "v_data.npy")
    q = np.load(pre_file + "q_data.npy")
    psi = np.load(pre_file + "psi_data.npy")
    
    return postprocess_mu_helper(u[last_n_outputs:-1, :, :], v[last_n_outputs:-1, :, :], 
                          q[last_n_outputs:-1, :, :], psi[last_n_outputs:-1, :, :], 
                          L, beta1, beta2, last_n_outputs)


# 8 args
def postprocess_mu_pyqg(pre_file, file_name, start, end, step, L, beta1, beta2):
    
    u, v, q, psi = load_netcdf(pre_file, file_name, start, end, step)
    
    return postprocess_mu_helper(u, v, 
                          q, psi, 
                          L, beta1, beta2)



def nummodel(q, yy, params, mu_c):
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


def nummodel_fft(q, yy, params, mu_c):
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


def nnmodel(torchmodel, omega, tau, dy):

    d_omega = gradient_first_c2f(omega, dy)

    omega_f = interpolate_c2f(omega)
    input  = torch.from_numpy(np.stack((abs(omega_f), d_omega)).T.astype(np.float32))
    mu_f = -torchmodel(input).detach().numpy().flatten()
    mu_f[mu_f >= 0.0] = 0.0

    mu_f = scipy.ndimage.gaussian_filter1d(mu_f, 5)
    
    M = gradient_first_f2c(mu_f*(d_omega), dy)

    return M

    

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
        model = lambda q, yy, params : nummodel(q, yy, params, mu_c)
    elif MODEL == "nnmodel":
        mymodel = torch.load("visc.model")
        model = lambda omega, tau, dy : nnmodel(mymodel, omega, tau, dy)
    else:
        print("ERROR")


    yy, t_data, q_data = explicit_solve(model, q0, f, params, dt = dt, Nt = Nt, save_every = save_every)
    
    return yy, t_data, q_data
