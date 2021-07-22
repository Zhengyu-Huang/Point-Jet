from Utility import gradient_first
import numpy as np
import scipy.io
import scipy.ndimage
import matplotlib.pyplot as plt
from NeuralNet import *
from datetime import datetime



data_dir_384 = "../data/beta_1.0_Gamma_1.0_relax_0.16/"

def load_data(data_dir):
    closure = scipy.io.loadmat(data_dir+"data_closure_cons.mat")["data_closure_cons"]
    dw_dy = scipy.io.loadmat(data_dir+"data_dw_dy.mat")["data_dw_dy"]
    dq_dy = scipy.io.loadmat(data_dir+"data_dq_dy.mat")["data_dq_dy"]
    w = scipy.io.loadmat(data_dir+"data_w.mat")["data_w"]
    q = scipy.io.loadmat(data_dir+"data_q.mat")["data_q"]
    return closure, dw_dy, dq_dy, w, q

closure, dw_dy, dq_dy, w, q = load_data(data_dir_384)
_, Ny, Nf = closure.shape

q_mean = np.mean(q[0, :, Nf//2:], axis=1)
q_mean_abs = np.abs(np.mean(q[0, :, Nf//2:], axis=1))
dq_dy_mean = np.mean(dq_dy[0, :, Nf//2:], axis=1)

# q_mean and dq_dy_mean



# dy = 4*np.pi/(Ny - 1)
# dq_dy_mean = gradient_first(q_mean, dy)
closure_mean = np.mean(closure[0, :, Nf//2:], axis=1)
mu_c = closure_mean/dq_dy_mean
mu_c[mu_c >= 0 ] = 0.0
mu_c[mu_c <= -0.1 ] = 0.0

mu_c = scipy.ndimage.gaussian_filter1d(mu_c, 5)
yy = np.linspace(-2*np.pi, 2*np.pi, Ny)

# chopping data
chop_l = 15
q_mean = q_mean[chop_l:-chop_l]
q_mean_abs = q_mean_abs[chop_l:-chop_l]
dq_dy_mean = dq_dy_mean[chop_l:-chop_l]
yy = yy[chop_l:-chop_l]
mu_c = mu_c[chop_l:-chop_l]

x_train  = torch.from_numpy(np.stack((q_mean, dq_dy_mean)).T.astype(np.float32))
y_train = torch.from_numpy(mu_c[:,np.newaxis].astype(np.float32))


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
	y_pred = -torch.square(model(x_train)) #-model(x_train) 
	loss = loss_fn(y_pred,y_train)*1000.0

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	if epoch % 100 == 0:
		print("[{}/{}], loss: {}, time {}".format(epoch, n_epochs, np.round(loss.item(), 3),datetime.now()))
		torch.save(model, "visc.model")

y_pred_train = -model(torch.from_numpy(np.stack((q_mean, dq_dy_mean)).T.astype(np.float32))).detach().numpy().flatten()**2

# plot data
plt.figure()
plt.plot(q_mean, dq_dy_mean, '--o', fillstyle="none", markevery = 1, markersize = 3, label="q")
plt.xlabel("q")
plt.ylabel("dq_dy")
plt.show()



plt.figure()
# plt.plot(yy, closure_mean/dq_dy_mean, '--o', fillstyle="none", markevery = 1, markersize = 3, label="mu")
plt.plot(yy, q_mean, '--o', fillstyle="none", markevery = 1, markersize = 3, label="q")
plt.plot(yy, dq_dy_mean, '--o', fillstyle="none", markevery = 1, markersize = 3, label="dq_dy")
plt.plot(yy, 100*mu_c,  '--o', fillstyle="none", markevery = 1, markersize = 3, label="filtered mu")
plt.plot(yy, 100*y_pred_train,  '--o', fillstyle="none", markevery = 1, markersize = 3, label="NN(q, dq_dy)")

# plt.xlim([-2,2])
plt.legend()
plt.xlabel("y")
plt.show()
plt.savefig("NN-fit.pdf")
# traced_fn = torch.jit.trace(model , (torch.rand(1, 1),))

# traced_fn = torch.jit.script(model)
#my_script_module = torch.jit.script

# omega = np.zeros(100, dtype="float32")

# omega = np.zeros(100)

# model(torch.reshape(torch.tensor(omega, dtype=torch.float32), (100,1))).detach().numpy()