# The computational domain is [0, 1]
# solve d/dt(q) + d/dx( D(q) d/dx(q) ) = f(x)
# boundary conditions are periodic


import scipy.io
import scipy.ndimage
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
import matplotlib.pyplot as plt
import sys
sys.path.append('../Utility')

class Kernel_Info():
    def __init__(self, kernel_name, delta, alpha):
        self.kernel_name = kernel_name
        self.delta = delta
        self.alpha = alpha
        
              
# constant box potential kernel = 3/delta**3
def kernel_intg(kernel_name, delta, alpha, a, b):
    if kernel_name == "constant_box_potential_kernel":
        a0, b0 = min(a,delta), min(b, delta)
        intg = 3.0/delta**3 * (b0**(alpha+1) - a0**(alpha+1))/(alpha+1) 
    elif kernel_name == "non_integrable_kernel":
        assert(alpha >= 1.0)
        a0, b0 = min(a,delta), min(b, delta)
        intg = 2.0/delta**2 * (b0**alpha - a0**alpha)/alpha 

    else:
        print("Kernel : ", kernel_name, " has not implemented")
            
    return intg

def nonlocal_model(kernel_name, delta, alpha, q, yy, res):
    Ny = yy.size
    dy = yy[1] - yy[0]
    r = int(delta / dy)
    res[:] = 0.0
    for i in range(1, Ny-1):
        for m in range(1, r+2):
            # intg_{(m-1)h, mh} kernel(s)/s^alpha
            intg = kernel_intg(kernel_name, delta, alpha, (m-1)*dy, m*dy)
            # mirroring
            qi_p_m = q[i + m] if i+m<=Ny-1 else q[Ny-1] - q[2*(Ny-1)-(i+m)]  #((i+m)*dy)**2 *(1.0 - ((i+m)*dy)**2) #0.0
            qi_m_m = q[i - m] if i-m>=0    else q[0] -  q[m-i]               #((i-m)*dy)**2 *(1.0 - ((i-m)*dy)**2) #0.0
            res[i] += (qi_p_m - 2*q[i] + qi_m_m)/(m*dy)**alpha * intg 
        #     print("i = ", i, " m = ", m , " r = ", r , qi_p_m , q[i] , qi_m_m, (qi_p_m - 2*q[i] + qi_m_m)/(m*dy)**alpha, intg )
        # print(res[i])
    
# the model is a function: q,t ->  M(q)
# the solution points are at cell faces
def explicit_solve(kernel_info, f, dbc, dt = 1.0, Nt = 1000, save_every = 1, L = 1.0):
    
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

    res = np.zeros(Ny)

    for i in range(1, Nt+1): 
        nonlocal_model(kernel_info.kernel_name, kernel_info.delta, kernel_info.alpha, q, yy, res)
        q[1:Ny-1] += dt*(f[1:Ny-1] + res[1:Ny-1])
        
        if i%save_every == 0:
            q_data[i//save_every, :] = q
            t_data[i//save_every] = i*dt
            print(i, "max q", np.max(q))

    return  yy, t_data, q_data


if __name__ == "__main__":
    Ny = 100
    L = 1.0
    yy, dy = np.linspace(0, L, Ny), L/(Ny - 1)
    delta = 0.1
    f = 12*yy**2 - 2.0  + 6*delta**2/5.0
    
    kernel_name = "constant_box_potential_kernel"
    q_ref = yy**2 *(1 - yy**2)
    kernel_info = Kernel_Info(kernel_name=kernel_name, delta=delta, alpha=0.0)
    dbc = [0.0, 0.0]
    
    #### TEST
    # res = np.zeros(Ny)
    # nonlocal_model(kernel_name, delta, 0.0, q_ref, yy, res)
    
    _, t_data, q_data = explicit_solve(kernel_info, f, dbc, dt = 1e-5, Nt = 50000, save_every = 1, L = 1.0)

    plt.figure()
    plt.plot(yy, q_ref, "--o", label="Ref")
    plt.plot(yy, q_data[-1, :], "--o", label="Pred")
    plt.legend()
    plt.show()


