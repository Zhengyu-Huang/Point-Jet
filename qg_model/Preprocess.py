import numpy as np
from Utility import gradient_first
import matplotlib.pyplot as plt

L = 2*np.pi
beta = 5.0


pre_file = "/Users/huang/Desktop/Code/GeophysicalFlows.jl/examples/"
u = np.load(pre_file + "u_data.npy")
v = np.load(pre_file + "v_data.npy")
q = np.load(pre_file + "q_data.npy")
psi = np.load(pre_file + "psi_data.npy")
nt, nx, ny, nlayers = u.shape


q_zonal_mean = np.mean(q, axis = 1)
dq_zonal_mean = np.copy(q_zonal_mean)



yy, dy = np.linspace(0, L, ny), L/(ny - 1)

v_zonal_mean = np.mean(v, axis = 1)
flux_zonal_mean = np.mean(v * q, axis = 1)
for i in range(nt):
    for j in range(nlayers):
        dq_zonal_mean[i, :, j] = gradient_first(q_zonal_mean[i, :, j], dy)


dpv_zonal_mean =  dq_zonal_mean + beta

t_mean_steps = range(-100,-1)
flux_mean    = np.mean(flux_zonal_mean[t_mean_steps, :, :], axis = 0)
dpv_mean     = np.mean(dpv_zonal_mean[t_mean_steps, :, :], axis = 0)
q_mean       = np.mean(q_zonal_mean[t_mean_steps, :, :], axis = 0)

# clipping 
dpv_mean[np.logical_and(dpv_mean >=-0.1 , dpv_mean <= 0.0)] = -0.1
dpv_mean[np.logical_and(dpv_mean <= 0.1 , dpv_mean >= 0.0)] =  0.1

mu_mean = flux_mean / dpv_mean

plt.plot(flux_mean[:, 0], yy, label="flux")
plt.plot(mu_mean[:, 0], yy, label="mu")
plt.plot(dpv_mean[:, 0], yy, label="dpv")
plt.legend()
plt.show()