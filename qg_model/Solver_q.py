# The computational domain is [-L/2, L/2]
# solve dq/dt + M(q, dq, ...) = (q_jet - q)/t
# boundary conditions depend on the modeling term (q = q_jet at top/bottom)





# solve dw/dt + M(w) = (w_jet - w)/t 
# boundary conditions depend on the modeling term 
# At top/bottom q is populated as q_jet or q_in depending on b(y)
import scipy.io
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Utility import gradient_first, gradient_second, gradient_first_c2f, gradient_first_f2c, interpolate_c2f, interpolate_f2c, psi_fft_sol



    
# the model is a function: w,t ->  M(w)
def explicit_solve(model, q0, f, params, dt = 1.0, Nt = 1000, save_every = 1):
    L, dU, beta, mu, F1, F2 = params["L"], params["dU"], params["beta"], params["mu"], params["F1"], params["F2"]
    
    _, Ny = q0.shape
    yy, dy = np.linspace(0, L, Ny), L/(Ny - 1)
    
    t = 0.0
    
    q = np.copy(q0)
    
    # prepare data storage
    q_data = np.zeros((Nt//save_every+1, 2, Ny))
    t_data = np.zeros(Nt//save_every+1)
    q_data[0, :, :], t_data[0] = q, t


    for i in range(1, Nt+1): 
        psi = psi_fft_sol(q, F1, F2, dy)
        dd_psi2 = gradient_second(psi[1, :], dy)
        q += dt*(f - model(q, yy, params))
        q[1, :] -= dt*mu*dd_psi2
        
        if i%save_every == 0:
            q_data[i//save_every, :, :] = q
            t_data[i//save_every] = i*dt
            print(i, "max q", np.max(q))

    return  yy, t_data, q_data

def postprocess_mu(pre_file, L, beta, last_n_outputs = 100):
    
    u = np.load(pre_file + "u_data.npy")
    v = np.load(pre_file + "v_data.npy")
    q = np.load(pre_file + "q_data.npy")
    psi = np.load(pre_file + "psi_data.npy")
    nt, nx, ny, nlayers = u.shape


    q_zonal_mean = np.mean(q, axis = 1)
    dq_zonal_mean = np.copy(q_zonal_mean)

    yy, dy = np.linspace(0, L, ny), L/(ny - 1)

    flux_zonal_mean = np.mean(v * q, axis = 1)
    for i in range(nt):
        for j in range(nlayers):
            dq_zonal_mean[i, :, j] = gradient_first(q_zonal_mean[i, :, j], dy)


    dpv_zonal_mean =  dq_zonal_mean + beta

    t_mean_steps = range(-last_n_outputs,-1)
    flux_mean    = np.mean(flux_zonal_mean[t_mean_steps, :, :], axis = 0)
    dpv_mean     = np.mean(dpv_zonal_mean[t_mean_steps, :, :], axis = 0)

    mu_mean = flux_mean / dpv_mean

    return mu_mean.T


def nummodel(q, yy, params):
    beta, dU, F1, F2, = params["beta"], params["dU"], params["F1"], params["F2"]
    dy = yy[1] - yy[0]
    
    mu_t = mu_c 
    q1, q2 = q[0, :], q[1, :]
    dq1, dq2 = gradient_first_f2c(q1, dy), gradient_first_f2c(q2, dy)
    
    # todo
    # mu_t[:,:] = 1e-3
    J1 = gradient_first_c2f(mu_t[0,:] * dq1, dy)
    J2 = gradient_first_c2f(mu_t[1,:] * dq2, dy)
    
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

    

Ny = 128
L = 12.8e6
f0, g = 1.0e-4, 10             # Coriolis parameter and gravitational constant
H = [5000, 5000]               # the rest depths of each layer
rho = [0.9, 1.0]               # the density of each layer
beta = 1.5e-11
params = {
    "L":    L,
    "dU":   5,
    "beta": beta,
    "mu":   1e-6,
    "F1":   f0**2/(g*(rho[1] - rho[0])/rho[1] * H[0]),
    "F2":   f0**2/(g*(rho[1] - rho[0])/rho[1] * H[1])
    }
# forcing
f = np.zeros((2, Ny))
yy = np.linspace(0, L, Ny)
q0 = np.zeros((2, Ny))
q0[0, :] = 1e-7 * np.sin(2*np.pi*yy/L)
q0[1, :] = 1e-7 * np.cos(2*np.pi*yy/L)
MODEL = "nummodel"

dt = 1800
Nt = 1000


if MODEL == "nummodel":
    pre_file = "/central/groups/esm/zhaoyi/geosphysicalflows_run/2layerqg/test1/"
    last_n_outputs = 100
    mu_mean = postprocess_mu(pre_file, L, beta, last_n_outputs)
    mu_c = np.zeros((2, Ny - 1))
    mu_c[0, :] = interpolate_f2c(mu_mean[0, :])
    mu_c[1, :] = interpolate_f2c(mu_mean[1, :])
    
    model = lambda q, yy, params : nummodel(q, yy, params)
elif MODEL == "nnmodel":
    mymodel = torch.load("visc.model")
    model = lambda omega, tau, dy : nnmodel(mymodel, omega, tau, dy)
else:
    print("ERROR")


yy, t_data, q_data = explicit_solve(model, q0, f, params, dt = dt, Nt = Nt, save_every = 1)
plt.figure()
plt.plot(np.mean(q_data[:, 0, :], axis=0), yy,  label="top")
plt.plot(np.mean(q_data[:, 1, :], axis=0), yy,  label="bottom")


plt.ylabel("y")
plt.legend()
plt.show()
 
