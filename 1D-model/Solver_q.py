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
            print(i, "max omega", max(omega))

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

    mu_t = mu_f 
    d_omega = gradient_first_c2f(omega, dy)
    M = gradient_first_f2c(mu_t*(d_omega), dy)


    return M


def nnmodel(torchmodel, omega, tau, dy):
    # return np.zeros(len(omega))
    # a0, a1, b0, b1 = 0.2, 0.3, 0.4, 0.5


    d_omega = gradient_first_c2f(omega, dy)
    omega_f = interpolate_c2f(omega)

    input  = torch.from_numpy(np.stack((omega_f, d_omega)).T.astype(np.float32))
    mu_f = -torchmodel(input).detach().numpy().flatten()**2
    # plt.figure()
    # plt.plot(mu_f)
    # plt.show()
    # mu_f[mu_f < -0.1] = -0.1
    M = gradient_first_f2c(mu_f*(d_omega), dy)

    return M

    

N = 384
omega_jet = np.zeros(N)
omega_jet[0:N//2] = 1.0
omega_jet[N-N//2:N] = -1.0
L = 4*np.pi
yy = np.linspace(-L/2.0, L/2.0, N)
omega_jet += 1*yy


#model = lambda omega, tau, dy : nnmodel(DirectNet_20(1, 1), omega, tau, dy)

data_dir = "../data/beta_1.0_Gamma_1.0_relax_0.16/"
dq_dy = scipy.io.loadmat(data_dir+"data_dq_dy.mat")["data_dq_dy"]
closure = scipy.io.loadmat(data_dir+"data_closure_cons.mat")["data_closure_cons"]
w = scipy.io.loadmat(data_dir+"data_w.mat")["data_w"]
q = scipy.io.loadmat(data_dir+"data_q.mat")["data_q"]

tau = 1/0.16
MODEL = "nnmodel"

if MODEL == "nummodel":
    _, Ny, Nf = closure.shape
    dy = L/(Ny - 1)
    q_mean = np.mean(q[0, :, Nf//2:], axis=1)
    dq_mean_dy = gradient_first(np.mean(q[0, :, Nf//2:], axis=1), dy)
    closure_mean = np.mean(closure[0, :, Nf//2:], axis=1)
    closure_mean_dy = gradient_first(closure_mean, dy)
    mu_c = np.mean(closure[0, :, Nf//2:], axis=1)/dq_mean_dy
    # mu_c = np.mean(closure[0, :, Nf//2:], axis=1)/np.mean(dq_dy[0, :, Nf//2:], axis=1)
    # Filter
    mu_c[mu_c >=0] = 0.0
    
    mu_f = interpolate_c2f(mu_c)
    model  = nummodel
    
elif MODEL == "nnmodel":
    mymodel = torch.load("visc.model")
    model = lambda omega, tau, dy : nnmodel(mymodel, omega, tau, dy)
else:
    print("ERROR")

yy, t_data, omega_data = explicit_solve(model, tau, omega_jet, dt = 0.001, Nt = 500000, save_every = 100, L = 4*np.pi)

plt.plot(yy, np.mean(omega_data, axis=0) - yy,  label="plug-in")
# plt.plot(yy, np.mean(omega_data, axis=0),  label="nn")
plt.plot(yy, np.mean(q[0,:,:].T, axis=0) - yy,  label="truth")
plt.xlabel("y")
# plt.xlim([-3,3])
plt.legend()
plt.show()
# animate(yy, t_data, omega_data)

# dy = yy[1] - yy[0]
# omega = omega_data[-1, :]
# d_omega = gradient_first_c2f(omega, dy)
# omega_f = interpolate_c2f(omega)
# input  = torch.from_numpy(np.stack((omega_f, d_omega)).T.astype(np.float32))
# mu_f = -mymodel(input).detach().numpy().flatten()**2
# plt.plot(omega_f)
# plt.plot(d_omega)
# plt.plot(mu_f)
# plt.show()

# plt.plot((omega_jet - q_mean)/tau , label="(q_jet - q) / tau")
# plt.plot(closure_mean_dy , label="dM_dy")
# plt.legend()
# plt.show()