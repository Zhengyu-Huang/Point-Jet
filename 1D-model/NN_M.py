from Utility import gradient_first
import numpy as np
import scipy.io
import scipy.ndimage
import matplotlib.pyplot as plt
from NeuralNet import *
from datetime import datetime
from Utility import gradient_first, gradient_second, gradient_first_c2f, gradient_first_f2c, interpolate_c2f
from NeuralNet import *


def load_data(data_dir):
	closure = scipy.io.loadmat(data_dir+"data_closure_cons.mat")["data_closure_cons"]
	dw_dy = scipy.io.loadmat(data_dir+"data_dw_dy.mat")["data_dw_dy"]
	dq_dy = scipy.io.loadmat(data_dir+"data_dq_dy.mat")["data_dq_dy"]
	w = scipy.io.loadmat(data_dir+"data_w.mat")["data_w"]
	q = scipy.io.loadmat(data_dir+"data_q.mat")["data_q"]
	return closure, dw_dy, dq_dy, w, q

PROCESS = "TEST"

if PROCESS == "TRAIN":

	data_dir_384 = "../data/beta_1.0_Gamma_1.0_relax_0.08/"
	closure, dw_dy, dq_dy, w, q = load_data(data_dir_384)
	_, Ny, Nf = closure.shape

	q_mean = np.mean(q[0, :, Nf//2:], axis=1)
	q_mean_abs = np.abs(np.mean(q[0, :, Nf//2:], axis=1))
	dq_dy_mean = np.mean(dq_dy[0, :, Nf//2:], axis=1)

	# dy = 4*np.pi/(Ny - 1)
	# dq_dy_mean = gradient_first(q_mean, dy)
	closure_mean = np.mean(closure[0, :, Nf//2:], axis=1)
	yy = np.linspace(-2*np.pi, 2*np.pi, Ny)

	# chopping data
	chop_l = 15
	closure_mean = closure_mean[chop_l:-chop_l]
	q_mean = q_mean[chop_l:-chop_l]
	q_mean_abs = q_mean_abs[chop_l:-chop_l]
	dq_dy_mean = dq_dy_mean[chop_l:-chop_l]
	yy = yy[chop_l:-chop_l]

	x_train  = torch.from_numpy(np.stack((q_mean_abs, dq_dy_mean)).T.astype(np.float32))
	y_train = torch.from_numpy(closure_mean[:,np.newaxis].astype(np.float32))


	# q_all = q[0, :, :].flatten()
	# dq_dy_all = dq_dy[0, :, :].flatten()
	# closure_all = closure[0, :, :].flatten()
	# mu_c_all = closure_all/dq_dy_all
	# x_train  = torch.from_numpy(np.stack((q_all, dq_dy_all)).T.astype(np.float32))
	# y_train = torch.from_numpy(mu_c_all[:,np.newaxis].astype(np.float32))


	ind = 2
	outd = 1 
	layers = 2
	width = 20
	# activation='relu'
	activation='tanh'
	model  = FNN(ind, outd, layers, width, activation)
	loss_fn = torch.nn.MSELoss(reduction='sum')
	learning_rate = 1e-3
	optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-4)
	n_epochs = 200000

	# model = torch.load("visc.model")

	for epoch in range(n_epochs):
		y_pred = model(x_train) 
		loss = loss_fn(y_pred,y_train)*1000.0

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if epoch % 100 == 0:
			print("[{}/{}], loss: {}, time {}".format(epoch, n_epochs, np.round(loss.item(), 3),datetime.now()))
			torch.save(model, "visc_M.model")

	y_pred_train = model(torch.from_numpy(np.stack((q_mean_abs, dq_dy_mean)).T.astype(np.float32))).detach().numpy().flatten()

	# plot data
	plt.figure()
	plt.plot(q_mean, dq_dy_mean, '--o', fillstyle="none", markevery = 1, markersize = 3, label="q")
	plt.xlabel("q")
	plt.ylabel("dq_dy")
	plt.show()



	plt.figure()

	plt.plot(yy, closure_mean,  '--o', fillstyle="none", markevery = 1, markersize = 3, label="closure_mean")
	plt.plot(yy, y_pred_train,  '--o', fillstyle="none", markevery = 1, markersize = 3, label="NN(q, dq_dy)")

	# plt.xlim([-2,2])
	plt.legend()
	plt.xlabel("y")
	plt.show()
	plt.savefig("NN-fit.pdf")





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


##########################################################################################
##########################################################################################
def plot_mean(yy, omega_data):
    plt.plot(np.mean(omega_data, axis=0), yy)
    


def nnmodel(torchmodel, omega, tau, dy):
    # return np.zeros(len(omega))
    # a0, a1, b0, b1 = 0.2, 0.3, 0.4, 0.5

	d_omega = gradient_first_c2f(omega, dy)

	omega_f = interpolate_c2f(omega)
	input  = torch.from_numpy(np.stack((abs(omega_f), d_omega)).T.astype(np.float32))
	M = torchmodel(input).detach().numpy().flatten()
	M = scipy.ndimage.gaussian_filter1d(M, 5)
	dM = gradient_first_f2c(M, dy)

	return dM

    
if PROCESS == "TEST":
	N = 384
	omega_jet = np.zeros(N)
	omega_jet[0:N//2] = 1.0
	omega_jet[N-N//2:N] = -1.0
	L = 4*np.pi
	yy = np.linspace(-L/2.0, L/2.0, N)
	omega_jet += 1*yy


	#model = lambda omega, tau, dy : nnmodel(DirectNet_20(1, 1), omega, tau, dy)

	tau_inv = "0.08"
	# tau_inv = "0.16"
	tau = 1/float(tau_inv)
	data_dir = "../data/beta_1.0_Gamma_1.0_relax_" + tau_inv + "/"

	closure, dw_dy, dq_dy, w, q = load_data(data_dir)

	_, Ny, Nf = closure.shape
	dy = L/(Ny - 1)

	mymodel = torch.load("visc_M.model")
	model = lambda omega, tau, dy : nnmodel(mymodel, omega, tau, dy)

	yy, t_data, omega_data = explicit_solve(model, tau, omega_jet, dt = 0.001, Nt = 200000, save_every = 100, L = 4*np.pi)
	plt.figure()
	plt.plot(yy, np.mean(omega_data, axis=0) - yy,  label="NN M")
	# plt.plot(yy, np.mean(omega_data, axis=0),  label="nn")
	plt.plot(yy, np.mean(q[0,:,:].T, axis=0) - yy,  label="truth")
	plt.xlabel("y")
	# plt.xlim([-3,3])
	plt.legend()
	plt.show()
