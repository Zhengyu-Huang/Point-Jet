import numpy as np
from scipy.fftpack import fft, ifft
from scipy.sparse.linalg import spsolve
from scipy import sparse
# periodic boundary condition without end point

# # extrapolate the gradient at the boundary
# def gradient_first(omega, dx):
#     nx = len(omega)
#     d_omega = np.copy(omega)
#     d_omega[1:nx-1] = (omega[2:nx] - omega[0:nx-2]) / (2*dx)
#     d_omega[0] = (omega[1] - omega[nx-1]) / (2*dx)
#     d_omega[nx-1] = (omega[0] - omega[nx-2]) / (2*dx)
#     return d_omega

# # extrapolate the gradient at the boundary
# def gradient_second(omega, dx):
#     nx = len(omega)
#     dd_omega = np.copy(omega)
#     dd_omega[1:nx-1] = (omega[2:nx] + omega[0:nx-2] - 2*omega[1:nx-1]) / (dx**2)
#     dd_omega[0] = (omega[1] + omega[nx-1] - 2*omega[0]) / (dx**2)
#     dd_omega[nx-1] = (omega[0] + omega[nx-2] - 2*omega[nx-1]) / (dx**2)
#     return dd_omega


# def gradient(omega, dx, order):
#     N = len(omega)
#     L = dx*N
#     k2= np.zeros(N)

#     if ((N%2)==0):
#         #-even number                                                                                   
#         for i in range(1,N//2):
#             k2[i]=i
#             k2[N-i]=-i
#     else:
#         #-odd number                                                                                    
#         for i in range(1,(N-1)//2):
#             k2[i]=i
#             k2[N-i]=-i
            
#     domega = np.copy(omega)
#     for i in range(order):
#         domega = 2*np.pi/L * ifft(1j*k2*fft(domega)).real
    
#     return domega


# compute gradient from face states to cell gradients, no boundary
def gradient_first_f2c(omega, dx):
    nx = len(omega)
    d_omega = (omega[1:nx] - omega[0:nx-1]) / (dx)
    return d_omega

# compute gradient from cell states to face gradients, no boundary
def gradient_first_c2f(omega, dx):
    nx = len(omega)
    d_omega = (omega[1:nx] - omega[0:nx-1]) / (dx)
    return d_omega


# compute gradient from cell states to face gradients, no boundary
def interpolate_f2c(omega):
    nx = len(omega)
    c_omega = (omega[0:nx-1] + omega[1:nx]) / 2.0
    return c_omega


# # compute gradient from cell states to face gradients
# # extrapolate the gradient at the boundary
# def interpolate_c2f(omega):
#     nx = len(omega)
#     f_omega = np.zeros(nx)
#     f_omega[1:nx] = (omega[1:nx] + omega[0:nx-1]) / 2.0
#     f_omega[0] = (omega[0] + omega[nx-1])/2.0
#     return f_omega






