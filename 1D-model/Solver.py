# The computational domain is [-L/2, L/2]
# solve dq/dt + M(q, dq, ...) = (q_jet - q)/t
# boundary conditions depend on the modeling term (q = q_jet at top/bottom)





# solve dw/dt + M(w) = (w_jet - w)/t 
# boundary conditions depend on the modeling term 
# At top/bottom q is populated as q_jet or q_in depending on b(y)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
from Utility import gradient_first, gradient_second

# the model is a function: w,t ->  M(w)
def explicit_solve(model, tau, omega_jet, dt = 1.0, Nt = 1000, save_every = 1, L = 4*np.pi):
    Ny = len(omega_jet)
    yy = np.linspace(-L/2.0, L/2.0, Ny)
    dy = L/(Ny - 1)

    t = 0.0
    omega = np.zeros(Ny)


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
    plt.show()


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



def model(omega, tau, dy):
    # return np.zeros(len(omega))
    # a0, a1, b0, b1 = 0.2, 0.3, 0.4, 0.5

    a0, a1, b0, b1 = -0.2, -0.3, -0.4, -0.5

    d_omega = gradient_first(omega, dy)

    return  ( a1*np.tanh(a0*d_omega + b0) + b1 ) / tau
    # return  ( a1*torch.relu( (a0*omega + b0)) + b1 ) / tau


# def model(torchmodel, omega, tau):
#     # return np.zeros(len(omega))
#     # a0, a1, b0, b1 = 0.2, 0.3, 0.4, 0.5

#     a0, a1, b0, b1 = -0.2, -0.3, -0.4, -0.5

#     # return  ( a1*np.tanh(a0*omega + b0) + b1 ) / tau
#     return  ( a1*torch.relu( torch.tensor(a0*omega + b0)) + b1 ) / tau


tau = 10.0
N = 384
omega_jet = np.zeros(N)
omega_jet[0:N//2] = 1.0
omega_jet[N-N//2:N] = -1.0


yy, t_data, omega_data = explicit_solve(model, tau, omega_jet, dt = 1.0, Nt = 100, save_every = 1, L = 4*np.pi)
# plot_mean(yy, omega_data)
animate(yy, t_data, omega_data)