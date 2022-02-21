# The computational domain is [-L/2, L/2]
# solve dq/dt + M(q, dq, ...) = (q_jet - q)/t
# boundary conditions depend on the modeling term (q = q_jet at top/bottom)


# solve dw/dt + M(w) = (w_jet - w)/t 
# boundary conditions depend on the modeling term 
# At top/bottom q is populated as q_jet or q_in depending on b(y)
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
# from Utility import gradient_first, gradient_second, gradient_first_c2f, gradient_first_f2c, interpolate_c2f
import sys
sys.path.append('../Utility')
from Numerics import gradient_first_c2f, gradient_first_f2c, interpolate_f2c


#########
#
#########

mu_scale = 1.0 #0.01
def create_net(ind, outd, layers, width, activation, initializer, outputlayer, params):

    net = NeuralNet.FNN(ind, outd, layers, width, activation, initializer, outputlayer) 
    net.update_params(params)
    return net

def net_eval(x, net):
    mu = net(torch.tensor(x, dtype=torch.float32)).detach().numpy().flatten() * mu_scale
    # data (prediction) clean 
    mu[mu <= 0.0] = 0.0
    mu = scipy.ndimage.gaussian_filter1d(mu, 5)
    return mu

def nn_flux(net, q, dq):
    x = np.vstack((q, dq)).T
    
    mu = net_eval(x, net) 
    return mu*dq





def load_data(data_dir):
    
    closure = -scipy.io.loadmat(data_dir+"data_closure_cons.mat")["data_closure_cons"]
    dq_dy   =  scipy.io.loadmat(data_dir+"data_dq_dy.mat")["data_dq_dy"]
    q       =  scipy.io.loadmat(data_dir+"data_q.mat")["data_q"]
    
    _, Ny, Nt = closure.shape

    q_mean = np.mean(q[0, :, Nt//2:], axis=1)
    dq_dy_mean = np.mean(dq_dy[0, :, Nt//2:], axis=1)
    closure_mean = np.mean(closure[0, :, Nt//2:], axis=1)

    return closure_mean, q_mean, dq_dy_mean







# the model is a function: w,t ->  M(w)
def explicit_solve(model, q_jet, tau, dt = 1.0, Nt = 1000, save_every = 1, L = 4*np.pi):
    
    Ny = q_jet.size
    yy = np.linspace(-L/2.0, L/2.0, Ny)
    dy = L/(Ny - 1)

    t = 0.0
    # q has Dirichlet boundary condition 
    q = np.copy(q_jet)

    q_data = np.zeros((Nt//save_every+1, Ny))
    t_data = np.zeros(Nt//save_every+1)
    q_data[0, :], t_data[0] = q, t

    
    res = np.zeros(Ny - 2)

    for i in range(1, Nt+1): 
        model(q, yy, res)

        # (q^{n+1} - q^n)/dt = res + (q_jet - q^{n+1})/tau
        q[1:Ny-1] = dt*tau/(dt + tau)*(q_jet[1:Ny-1]/tau + res + q[1:Ny-1]/dt)

        if i%save_every == 0:
            q_data[i//save_every, :] = q
            t_data[i//save_every] = i*dt
            print(i, "max q", np.max(q))

    return  yy, t_data, q_data

# def plot_mean(yy, q_data):
#     plt.plot(np.mean(q_data, axis=0), yy)
    
# def animate(yy, t_data, q_data):
#     fig = plt.figure()
#     ax = plt.axes(xlim=(q_data.min(), q_data.max()), ylim=(yy.min(), yy.max()))
#     line, = ax.plot([], [], lw=2)

#     # initialization function: plot the background of each frame
#     def init():
#         line.set_data(q_data[0, :], yy)
#         return line,

#     # animation function.  This is called sequentially
#     def animate(i):
#         x = q_data[i, :]    
#         y = yy
#         line.set_data(x, y)
#         return line,

#     # call the animator.  blit=True means only re-draw the parts that have changed.
#     anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=200, interval=20, blit=True)
#     # anim.save('q.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

#     plt.show()



def nummodel(permeability, q, yy, res):

    Ny = yy.size
    dy = yy[1] - yy[0]
    dq_c = gradient_first_f2c(q, dy)
    q_c = interpolate_f2c(q)
    mu_c = permeability(q_c, dq_c)
    
    
    # mu_c[mu_t >=0] = 0.0
    # mu_c[mu_t <=-0.1] = 0.0


    res[:] = gradient_first_c2f(mu_c*(dq_c), dy)


# dM/dx    
def nummodel_flux(flux, q, yy, res):
    
    Ny = yy.size
    dy = yy[1] - yy[0]
    dq_c = gradient_first_f2c(q, dy)
    q_c = interpolate_f2c(q)
    M_c = flux(q_c, dq_c)
    res[:] = gradient_first_c2f(M_c, dy)



# def nnmodel(torchmodel, q, tau, dy):
#     # return np.zeros(len(q))
#     # a0, a1, b0, b1 = 0.2, 0.3, 0.4, 0.5
    
#     d_q = gradient_second(q, dy)

#     return torchmodel(torch.reshape(torch.tensor(d_q, dtype=torch.float32), (len(q),1))).detach().numpy().flatten() / tau

#     # return  ( a1*np.tanh(a0*q + b0) + b1 ) / tau
#     # return  ( a1*torch.relu( torch.tensor(a0*q + b0)) + b1 ) / tau


# tau = 10.0
# N = 384
# q_jet = np.zeros(N)
# q_jet[0:N//2] = 1.0
# q_jet[N-N//2:N] = -1.0


# #model = lambda q, tau, dy : nnmodel(DirectNet_20(1, 1), q, tau, dy)

# data_dir = "../data/beta_1.0_Gamma_1.0_relax_0.16/"
# dq_dy = scipy.io.loadmat(data_dir+"data_dq_dy.mat")["data_dq_dy"]
# closure = scipy.io.loadmat(data_dir+"data_closure_cons.mat")["data_closure_cons"]
# w = scipy.io.loadmat(data_dir+"data_w.mat")["data_w"]

# _, _, Nf = closure.shape
# mu_c = np.mean(closure[0, :, Nf//2:], axis=1)/np.mean(dq_dy[0, :, Nf//2:], axis=1)


# mu_c[mu_c >=0] = 0.0
# mu_c[mu_c <=-0.1] = 0.0


# mu_f = interpolate_c2f(mu_c)

# model  = nummodel
# # 50000
# yy, t_data, q_data = explicit_solve(model, tau, q_jet, dt = 0.001, Nt = 500000, save_every = 100, L = 4*np.pi)

# plt.plot(np.mean(q_data, axis=0), yy, label="fit")
# plt.plot(np.mean(w[0,:,:].T, axis=0)+yy, yy, label="truth")
# plt.plot(q_jet, yy, label="jet")
# plt.plot(mu_c, yy, label="mu")
# plt.legend()
# plt.show()
# # animate(yy, t_data, q_data)

# plt.plot(np.mean(dq_dy[0, :, Nf//2:], axis=1), yy, "-o", fillstyle="none", label="dq_dy")
# plt.plot(mu_c, yy, "-o", fillstyle="none", label="mu_c")
# plt.plot(np.mean(closure[0, :, Nf//2:], axis=1), yy, "-o", fillstyle="none", label="closure")
# plt.legend()
# plt.show()