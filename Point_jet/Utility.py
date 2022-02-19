import numpy as np

# extrapolate the gradient at the boundary
def gradient_first(omega, dx):
    nx = len(omega)
    d_omega = np.copy(omega)
    d_omega[1:nx-1] = (omega[2:nx] - omega[0:nx-2]) / (2*dx)
    d_omega[0], d_omega[nx-1] = d_omega[1], d_omega[nx-2]
    return d_omega

# extrapolate the gradient at the boundary
def gradient_second(omega, dx):
    nx = len(omega)
    dd_omega = np.copy(omega)
    dd_omega[1:nx-1] = (omega[2:nx] + omega[0:nx-2] - 2*omega[1:nx-1]) / (dx**2)
    dd_omega[0], dd_omega[nx-1] = dd_omega[1], dd_omega[nx-2]
    return dd_omega




# compute gradient from face states to cell gradients
def gradient_first_f2c(omega, dx):
    nx = len(omega) - 1
    d_omega = (omega[1:nx+1] - omega[0:nx]) / (dx)
    return d_omega

# compute gradient from cell states to face gradients
# extrapolate the gradient at the boundary
def gradient_first_c2f(omega, dx):
    nx = len(omega)
    d_omega = np.zeros(nx+1)
    d_omega[1:nx] = (omega[1:nx] - omega[0:nx-1]) / (dx)
    d_omega[0], d_omega[nx] = d_omega[1], d_omega[nx-1]
    return d_omega


# compute gradient from cell states to face gradients
# extrapolate the gradient at the boundary
def interpolate_c2f(omega):
    nx = len(omega)
    f_omega = np.zeros(nx+1)
    f_omega[1:nx] = (omega[1:nx] + omega[0:nx-1]) / 2.0
    f_omega[0], f_omega[nx] = omega[0], omega[nx-1]
    return f_omega