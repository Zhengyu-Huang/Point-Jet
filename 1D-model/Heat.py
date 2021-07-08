import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Utility import gradient_first, gradient_second
from NeuralNet import *


# generate data for heat equation

def heat_solve(tau, omega_jet, dt = 1.0, Nt = 1000, save_every = 1, L = 4*np.pi):
    nu = 0.1
    omega_top, omega_bottom = 0.0, 1.0

    Ny = len(omega_jet)
    yy = np.linspace(-L/2.0, L/2.0, Ny)
    dy = L/(Ny - 1)

    t = 0.0
    omega = np.zeros(Ny)
    omega = np.copy(omega_jet)

    omega = np.linspace(omega_bottom, omega_top, Ny)

    omega_data = np.zeros((Nt//save_every+1, Ny))
    t_data = np.zeros(Nt//save_every+1)
    omega_data[0, :], t_data[0] = omega, t
    M1 = np.zeros((Nt//save_every+1, Ny))
    M2 = np.zeros((Nt//save_every+1, Ny))

    for i in range(1, Nt+1): 
        d_omega = gradient_first(omega, dy)
        dd_omega = gradient_second(omega, dy)
        

        omega[1:-1] = dt*tau/(dt + tau)*(omega_jet[1:-1]/tau + nu*dd_omega[1:-1] + omega[1:-1]/dt)

        omega[-1], omega[0] = omega_top, omega_bottom

        if i%save_every == 0:
            omega_data[i//save_every, :] = omega
            M1[i//save_every, :] = nu*dd_omega
            M2[i//save_every, :] = nu*d_omega 

            t_data[i//save_every] = i*dt
    return  yy, t_data, omega_data, M1, M2

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



L = 4*np.pi
N = 384
yy = np.linspace(-L/2.0, L/2.0, N)
omega_jet = np.zeros(N)

# parameters
tau = 20.0
Gamma = 1.0
jet_type = "Point"
jet_type = "Gaussian"


if jet_type == "Point":
    omega_jet[0:N//2] = Gamma
    omega_jet[N-N//2:N] = -Gamma
elif jet_type == "Gaussian":
    jet_width = Gamma
    omega_jet = np.exp(-yy**2/(2*jet_width**2))
# else if jet_type == "Double-Gaussian":
#     jet_width = 1.0
#     omega_jet = -(yy)/jet_width^2*exp(-(yy + L/4)^2/jet_width^2/2) - (y - L/4)/jet_width^2*exp(-(yy - L/4)^2/jet_width^2/2)



#model = lambda omega, tau, dy : nnmodel(DirectNet_20(1, 1), omega, tau, dy)


yy, t_data, omega_data, M1, M2 = heat_solve(tau, omega_jet, dt = 0.005, Nt = 100000, save_every = 100, L = L)
# plot_mean(yy, omega_data)
# animate(yy, t_data, omega_data)

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(15,6))
fig.suptitle("Jet type = %s Gamma=%1.1f tau=%1.1f" %(jet_type, Gamma, tau))
ax1.plot(omega_jet, yy)
ax1.set_ylabel("omega_jet")
ax2.plot(omega_data[-1, :], yy)
ax2.set_ylabel("omega")
ax3.plot(M1[-1, :], yy)
ax3.set_ylabel("M1")
ax4.plot(M2[-1, :], yy)
ax4.set_ylabel("M2")
ax5.plot(omega_data[-1, :], M1[-1, :], "o", fillstyle="none")
ax5.plot(omega_data[-1, :], -(omega_jet-omega_data[-1, :])/tau , "-r")
ax5.set_xlabel("omega")
ax5.set_ylabel("M1")
fig.tight_layout()
plt.savefig("Jet type = %s Gamma=%1.1f tau=%1.1f.png" %(jet_type, Gamma, tau))
animate(yy, t_data, omega_data)