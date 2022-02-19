import numpy as np
from scipy.fftpack import fft, ifft
from scipy.sparse.linalg import spsolve
from scipy import sparse
# periodic boundary condition!

# extrapolate the gradient at the boundary
def gradient_first(omega, dx):
    nx = len(omega)
    d_omega = np.copy(omega)
    d_omega[1:nx-1] = (omega[2:nx] - omega[0:nx-2]) / (2*dx)
    d_omega[0] = d_omega[nx-1] = (omega[1] - omega[nx-2]) / (2*dx)
    return d_omega

# extrapolate the gradient at the boundary
def gradient_second(omega, dx):
    nx = len(omega)
    dd_omega = np.copy(omega)
    dd_omega[1:nx-1] = (omega[2:nx] + omega[0:nx-2] - 2*omega[1:nx-1]) / (dx**2)
    dd_omega[0] = dd_omega[nx-1] = (omega[1] + omega[nx-2] - 2*omega[0]) / (dx**2)
    return dd_omega


def gradient(omega, dx, order):
    nx = len(omega)
    N = nx - 1 # open periodic domain
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
            
    domega = omega[0:-1]
    for i in range(order):
        domega = 2*np.pi/L * ifft(1j*k2*fft(domega)).real
    
    return np.append(domega, domega[0])


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
    d_omega[0] = d_omega[nx] = (omega[0] - omega[nx-1]) / (dx)
    return d_omega


# compute gradient from cell states to face gradients
# extrapolate the gradient at the boundary
def interpolate_c2f(omega):
    nx = len(omega)
    f_omega = np.zeros(nx+1)
    f_omega[1:nx] = (omega[1:nx] + omega[0:nx-1]) / 2.0
    f_omega[0] = f_omega[nx] = (omega[0] + omega[nx-1])/2.0
    return f_omega


# compute gradient from cell states to face gradients
# extrapolate the gradient at the boundary
def interpolate_f2c(omega):
    nx = len(omega)
    c_omega = np.zeros(nx-1)
    c_omega = (omega[1:nx] + omega[0:nx-1]) / 2.0
    return c_omega




# solve psi from q by FD
# q_1 = dd psi_1 + F1(psi_2 - psi_1)
# q_2 = dd psi_2 + F2(psi_1 - psi_2)
def psi_fd_sol(q, F1, F2, dy):
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




# solve psi from q by FFT
# q_1 = dd psi_1 + F1(psi_2 - psi_1)
# q_2 = dd psi_2 + F2(psi_1 - psi_2)
def psi_fft_sol(q, F1, F2, dy):
    _, nx = q.shape 
    N = nx - 1 # open periodic domain
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


    psi[0, N] = psi[0, 0]
    psi[1, N] = psi[1, 0]
    
    return psi