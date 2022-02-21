import numpy as np
from scipy.fftpack import fft, ifft
from scipy.sparse.linalg import spsolve
from scipy import sparse
# periodic boundary condition without end point

# # extrapolate the gradient at the boundary
def gradient_first(omega, dx, bc = "one-sided"):
    nx = len(omega)
    d_omega = np.copy(omega)
    d_omega[1:nx-1] = (omega[2:nx] - omega[0:nx-2]) / (2*dx)
    
    if bc == "periodic":
        d_omega[0] = (omega[1] - omega[nx-1]) / (2*dx)
        d_omega[nx-1] = (omega[0] - omega[nx-2]) / (2*dx)
    else: #one-sided gradient 
        d_omega[0] = (omega[1] - omega[0]) / (dx)
        d_omega[nx-1] = (omega[nx-1] - omega[nx-2]) / (dx)
    return d_omega




# compute gradient from face states to cell gradients, no boundary
def gradient_first_f2c(omega, dx, bc = "None"):
    nx = len(omega)
    
    if bc == "periodic":
        d_omega = np.copy(omega)
        d_omega[0:nx-1] = (omega[1:nx] - omega[0:nx-1]) / (dx)
        d_omega[nx-1] = (omega[0] - omega[nx-1]) / (dx)
    else:
        d_omega = (omega[1:nx] - omega[0:nx-1]) / (dx)
    return d_omega



# compute gradient from cell states to face gradients, no boundary
def gradient_first_c2f(omega, dx, bc = "None"):
    nx = len(omega)
    
    if bc == "periodic":
        d_omega = np.zeros(nx)
        d_omega[1:nx] = (omega[1:nx] - omega[0:nx-1]) / (dx)
        d_omega[0] = (omega[0] - omega[nx-1]) / (dx)
    else:
        d_omega = (omega[1:nx] - omega[0:nx-1]) / (dx)
    return d_omega

# compute gradient from cell states to face gradients, no boundary
def interpolate_f2c(omega, bc = "None" ):
    nx = len(omega)
    
    if bc == "periodic":
        c_omega = np.zeros(nx)
        c_omega[0:nx-1] = (omega[0:nx-1] + omega[1:nx]) / 2.0
        c_omega[nx-1] = (omega[0] + omega[nx-1]) / 2.0
    else:
        c_omega = (omega[0:nx-1] + omega[1:nx]) / 2.0
    return c_omega


# compute gradient from cell states to face gradients
# extrapolate the gradient at the boundary
def interpolate_c2f(omega, bc = "None"):
    nx = len(omega)
    
    if bc == "periodic":
        f_omega = np.zeros(nx)
        f_omega[1:nx] = (omega[1:nx] + omega[0:nx-1]) / 2.0
        f_omega[0] = (omega[0] + omega[nx-1])/2.0
    else:
        f_omega[1:nx] = (omega[0:nx-1] + omega[1:nx]) / 2.0
    return f_omega







import numpy as np
from scipy.fftpack import fft, ifft
from scipy.sparse.linalg import spsolve
from scipy import sparse
# periodic boundary condition without end point



# # extrapolate the gradient at the boundary
# def gradient_second(omega, dx):
#     nx = len(omega)
#     dd_omega = np.copy(omega)
#     dd_omega[1:nx-1] = (omega[2:nx] + omega[0:nx-2] - 2*omega[1:nx-1]) / (dx**2)
#     dd_omega[0] = (omega[1] + omega[nx-1] - 2*omega[0]) / (dx**2)
#     dd_omega[nx-1] = (omega[0] + omega[nx-2] - 2*omega[nx-1]) / (dx**2)
#     return dd_omega


def gradient_fft(omega, dx, order):
    N = len(omega)
    L = dx*N
    k2= np.zeros(N)

    if ((N%2)==0):
        #-even number                                                                                   
        for i in range(1,N//2):
            k2[i]=i
            k2[N-i]=-i
    else:
        #-odd number                                                                                    
        for i in range(1,(N-1)//2):
            k2[i]=i
            k2[N-i]=-i
            
    domega = np.copy(omega)
    for i in range(order):
        domega = 2*np.pi/L * ifft(1j*k2*fft(domega)).real
    
    return domega



# solve psi from q by FFT
# q_1 = dd psi_1 + F1(psi_2 - psi_1)
# q_2 = dd psi_2 + F2(psi_1 - psi_2)
def psi_fft_sol(q, F1, F2, dy):
    _, N = q.shape 
    L = dy*N
    
    k2= np.zeros(N)

    if ((N%2)==0):
        #-even number                                                                                   
        for i in range(1,N//2):
            k2[i]   =  i
            k2[N-i] = -i
    else:
        #-odd number                                                                                    
        for i in range(1,(N-1)//2):
            k2[i]   =  i
            k2[N-i] = -i

    ddx = - (2*np.pi/L)**2 * k2**2

    q_h   = np.zeros((2, N), dtype=np.complex)
    psi_h = np.zeros((2, N), dtype=np.complex)
    psi = np.copy(q)

    q_h[0, :], q_h[1, :] = fft(q[0,0:N]), fft(q[1,0:N])

    for i in range(N):
        det = ddx[i]**2 - (F2 + F1)*ddx[i]
        if abs(det < 1e-10):
            psi_h[0,i], psi_h[1,i] = 0.0, 0.0
        else:
            psi_h[0,i] = ((ddx[i]-F2)*q_h[0,i] - F1*q_h[1,i])/det 
            psi_h[1,i] = (-F2*q_h[0,i] + (ddx[i]-F1)*q_h[1,i])/det


    psi[0, 0:N] = ifft(psi_h[0,:]).real
    psi[1, 0:N] = ifft(psi_h[1,:]).real
    
    return psi


# # solve psi from q by FD
# # q_1 = dd psi_1 + F1(psi_2 - psi_1)
# # q_2 = dd psi_2 + F2(psi_1 - psi_2)
# def psi_fd_sol(q, F1, F2, dy):
#     print("WARNING: Shift by a constant")
#     _, Ny = q.shape 
#     # right hand side
#     q_ext = np.zeros(2*Ny)
#     q_ext[0:Ny] = q[0, 0:Ny]
#     q_ext[Ny:2*Ny] = q[1, 0:Ny]
#     # sparse matrix
    
#     I, J, V = [], [], []
#     for i in range(Ny):
#         I.extend([i, i, i, i, i])
#         J.extend([(i-1)%(Ny), i, (i+1)%(Ny), i, i+Ny])
#         V.extend([1/dy**2, -2/dy**2, 1/dy**2, -F1, F1])
        
#         I.extend([i+Ny, i+Ny, i+Ny, i+Ny, i+Ny])
#         J.extend([(i-1)%(Ny)+Ny, i+Ny, (i+1)%(Ny)+Ny, i, i+Ny])
#         V.extend([1/dy**2, -2/dy**2, 1/dy**2, F2, -F2])
        
    
#     A = sparse.coo_matrix((V,(I,J)),shape=(2*(Ny),2*(Ny))).tocsr()
#     psi_ext = spsolve(A, q_ext)
    
#     psi = np.zeros((2, Ny))
#     psi[0, 0:Ny] = psi_ext[0:Ny]
#     psi[1, 0:Ny] = psi_ext[Ny:2*(Ny)] 
    
#     return psi











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