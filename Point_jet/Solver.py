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
from Utility import gradient_first, gradient_second, gradient_first_c2f, gradient_first_f2c, interpolate_c2f
from NeuralNet import *


# the model is a function: w,t ->  M(w)
def explicit_solve(model, tau, omega_jet, dt = 1.0, Nt = 1000, save_every = 1, L = 4*np.pi):
    Ny = len(omega_jet)
    yy = np.linspace(-L/2.0, L/2.0, Ny)
    dy = L/(Ny - 1)

    t = 0.0
    omega = np.zeros(Ny)
    omega = np.copy(omega_jet)

    omega_data = np.zeros((Nt//save_every+1, Ny))
    t_data = np.zeros(Nt//save_every+1)
    omega_data[0, :], t_data[0] = omega, t

    
    

    for i in range(1, Nt+1): 
        omega = dt*tau/(dt + tau)*(omega_jet/tau - model(omega, tau, dy) + omega/dt)

        # omega = omega + dt * model(omega, tau) 


        if i%save_every == 0:
            omega_data[i//save_every, :] = omega
            t_data[i//save_every] = i*dt

    return  yy, t_data, omega_data

def plot_mean(yy, omega_data):
    plt.plot(np.mean(omega_data, axis=0), yy)
    


def animate(yy, t_data, omega_data):
    fig = plt.figure()
    ax = plt.axes(xlim=(omega_data.min(), omega_data.max()), ylim=(yy.min(), yy.max()))
    line, = ax.plot([], [], lw=2)

    # initialization function: plot the background of each frame
    def init():
        line.set_data(omega_data[0, :], yy)
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        x = omega_data[i, :]    
        y = yy
        line.set_data(x, y)
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)
    # anim.save('omega.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()



def nummodel(omega, tau, dy):
    # return np.zeros(len(omega))
    # a0, a1, b0, b1 = 0.2, 0.3, 0.4, 0.5

    # a0, a1, b0, b1 = -0.2, -0.3, -0.4, -0.5

    # d_omega = gradient_second(omega, dy)
    # return  -1e-2*d_omega /tau # ( a1*np.tanh(a0*d_omega + b0) + b1 ) / tau

    # ind = omega >= 0.0
    # M = np.copy(omega)
    # M[ind] = 0.8 - 0.8*omega[ind]
    # M[~ind] = -0.8 - 0.8*omega[~ind]
    # return  ( a1*torch.relu( (a0*omega + b0)) + b1 ) / tau

    mu_t = mu_f 

    mu_t[mu_t >=0] = 0.0
    mu_t[mu_t <=-0.1] = 0.0


    d_omega = gradient_first_c2f(omega, dy)

    M = gradient_first_f2c(mu_t*(d_omega + 1), dy)


    # M = mu_c*gradient_second(omega, dy)

    # d_omega = gradient_first(omega, dy)
    # M = gradient_first(mu_t*d_omega, dy)
    
    return M


def nnmodel(torchmodel, omega, tau, dy):
    # return np.zeros(len(omega))
    # a0, a1, b0, b1 = 0.2, 0.3, 0.4, 0.5
    
    d_omega = gradient_second(omega, dy)

    return torchmodel(torch.reshape(torch.tensor(d_omega, dtype=torch.float32), (len(omega),1))).detach().numpy().flatten() / tau

    # return  ( a1*np.tanh(a0*omega + b0) + b1 ) / tau
    # return  ( a1*torch.relu( torch.tensor(a0*omega + b0)) + b1 ) / tau


tau = 10.0
N = 384
omega_jet = np.zeros(N)
omega_jet[0:N//2] = 1.0
omega_jet[N-N//2:N] = -1.0


#model = lambda omega, tau, dy : nnmodel(DirectNet_20(1, 1), omega, tau, dy)

data_dir = "../data/beta_1.0_Gamma_1.0_relax_0.16/"
dq_dy = scipy.io.loadmat(data_dir+"data_dq_dy.mat")["data_dq_dy"]
closure = scipy.io.loadmat(data_dir+"data_closure_cons.mat")["data_closure_cons"]
w = scipy.io.loadmat(data_dir+"data_w.mat")["data_w"]

_, _, Nf = closure.shape
mu_c = np.mean(closure[0, :, Nf//2:], axis=1)/np.mean(dq_dy[0, :, Nf//2:], axis=1)


mu_c[mu_c >=0] = 0.0
mu_c[mu_c <=-0.1] = 0.0


mu_f = interpolate_c2f(mu_c)

model  = nummodel
# 50000
yy, t_data, omega_data = explicit_solve(model, tau, omega_jet, dt = 0.001, Nt = 500000, save_every = 100, L = 4*np.pi)

plt.plot(np.mean(omega_data, axis=0), yy, label="fit")
plt.plot(np.mean(w[0,:,:].T, axis=0)+yy, yy, label="truth")
plt.plot(omega_jet, yy, label="jet")
plt.plot(mu_c, yy, label="mu")
plt.legend()
plt.show()
# animate(yy, t_data, omega_data)

plt.plot(np.mean(dq_dy[0, :, Nf//2:], axis=1), yy, "-o", fillstyle="none", label="dq_dy")
plt.plot(mu_c, yy, "-o", fillstyle="none", label="mu_c")
plt.plot(np.mean(closure[0, :, Nf//2:], axis=1), yy, "-o", fillstyle="none", label="closure")
plt.legend()
plt.show()