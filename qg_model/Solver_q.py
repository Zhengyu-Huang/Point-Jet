# The computational domain is [-L/2, L/2]
# solve dq/dt + M(q, dq, ...) = (q_jet - q)/t
# boundary conditions depend on the modeling term (q = q_jet at top/bottom)





# solve dw/dt + M(w) = (w_jet - w)/t 
# boundary conditions depend on the modeling term 
# At top/bottom q is populated as q_jet or q_in depending on b(y)
import scipy.io
import scipy.ndimage
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Utility import gradient_first, gradient_second, gradient_first_c2f, gradient_first_f2c, interpolate_c2f, interpolate_f2c
from NeuralNet import *

# solve psi from q
def psi_sol(q, F1, F2, dy):
    _, Ny = q.shape 
    # right hand side
    q_ext = np.zeros(2*(Ny - 1))
    q_ext[0:Ny-1] = q[0, 0:Ny-1]
    q_ext[Ny-1:2*(Ny - 1)] = q[1, 0:Ny-1]
    # sparse matrix
    
    I, J, V = [], [], []
    for i in range(Ny - 1):
        I.extend([i, i, i, i, i])
        J.extend([(i-1)%(Ny-1), i, (i+1)%(Ny-1), i, i+Ny-1])
        V.extend([1/dy**2, -2/dy**2, 1/dy**2, -F1, F1])
        
        I.extend([i+Ny-1, i+Ny-1, i+Ny-1, i+Ny-1, i+Ny-1])
        J.extend([(i-1)%(Ny-1)+Ny-1, i+Ny-1, (i+1)%(Ny-1)+Ny-1, i, i+Ny-1])
        V.extend([1/dy**2, -2/dy**2, 1/dy**2, F2, -F2])
        
    
    A = sparse.coo_matrix((V,(I,J)),shape=(2*(Ny - 1),2*(Ny - 1))).tocsr()
    psi_ext = spsolve(A, q_ext)
    
    psi = np.zeros((2, Ny))
    psi[0, 0:Ny-1] = psi_ext[0:Ny-1]
    psi[0, Ny-1] = psi[0, 0]
    psi[1, 0:Ny-1] = psi_ext[Ny-1:2*(Ny - 1)] 
    psi[1, Ny-1] = psi[1, 0]
    
    return psi
    
    
# the model is a function: w,t ->  M(w)
def explicit_solve(model, f, params, dt = 1.0, Nt = 1000, save_every = 1):
    L, dU, beta, mu, F1, F2 = params["L"], params["dU"], params["beta"], params["mu"], params["F1"], params["F2"]
    
    _, Ny = f.shape
    yy, dy = np.linspace(0, L, Ny), L/(Ny - 1)
    
    t = 0.0
    # qi are periodic (qi[0] == qi[-1])
    q = np.zeros((2, Ny))
    
    # prepare data storage
    q_data = np.zeros((Nt//save_every+1, 2, Ny))
    t_data = np.zeros(Nt//save_every+1)
    q_data[0, :, :], t_data[0] = q, t


    for i in range(1, Nt+1): 
        psi = psi_sol(q, F1, F2, dy)
        dd_psi2 = gradient_second(psi[1, :], dy)
        q += dt*(f - model(q, yy, params))
        q[1, :] -= dt*mu*dd_psi2
        
        if i%save_every == 0:
            q_data[i//save_every, :, :] = q
            t_data[i//save_every] = i*dt
            print(i, "max q", np.max(q))

    return  yy, t_data, q_data


def nummodel(q, yy, params):
    beta, dU, F1, F2, = params["beta"], params["dU"], params["F1"], params["F2"]
    dy = yy[1] - yy[0]
    
    mu_t = mu_c 
    q1, q2 = q[0, :], q[1, :]
    dq1, dq2 = gradient_first_f2c(q1, dy), gradient_first_f2c(q2, dy)
    
    J1 = gradient_first_c2f(mu_t*(dq1), dy)
    J2 = gradient_first_c2f(mu_t*(dq2), dy)

    return np.vstack((J1, J2))


def nnmodel(torchmodel, omega, tau, dy):
    # return np.zeros(len(omega))
    # a0, a1, b0, b1 = 0.2, 0.3, 0.4, 0.5

    d_omega = gradient_first_c2f(omega, dy)


    omega_f = interpolate_c2f(omega)
    input  = torch.from_numpy(np.stack((abs(omega_f), d_omega)).T.astype(np.float32))
    mu_f = -torchmodel(input).detach().numpy().flatten()
    mu_f[mu_f >= 0.0] = 0.0

    mu_f = scipy.ndimage.gaussian_filter1d(mu_f, 5)
    
    M = gradient_first_f2c(mu_f*(d_omega), dy)

    return M

    

Ny = 100
L = 2*np.pi
params = {
    "L": L,
    "dU":  1,
    "beta": 5,
    "mu": 5e-2,
    "F1": 1/((5.0-4.0)/5.0 * 0.2),
    "F2": 1/((5.0-4.0)/5.0 * 0.8)
    }
# forcing
f = np.zeros((2, Ny))
yy = np.linspace(0, L, Ny)
MODEL = "nummodel"


if MODEL == "nummodel":
    mu_f = np.sin(yy)
    mu_c = interpolate_f2c(mu_f)
    model = lambda q, yy, params : nummodel(q, yy, params)
    
elif MODEL == "nnmodel":
    mymodel = torch.load("visc.model")
    model = lambda omega, tau, dy : nnmodel(mymodel, omega, tau, dy)
else:
    print("ERROR")

yy, t_data, q_data = explicit_solve(model, f, params, dt = 1.0, Nt = 1000, save_every = 1)
plt.figure()
plt.plot(yy, np.mean(q_data[0, :], axis=0),  label="top")
plt.plot(yy, np.mean(q_data[1, :], axis=0),  label="bottom")


plt.xlabel("y")
plt.legend()
plt.show()
 
