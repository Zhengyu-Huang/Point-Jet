import numpy as np
import scipy.io
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
dq_dy_mean = np.mean(dq_dy[0, :, Nf//2:], axis=1)
closure_mean = np.mean(closure[0, :, Nf//2:], axis=1)

mu_c = closure_mean/dq_dy_mean




x_train  = torch.from_numpy(np.stack((q_mean, dq_dy_mean)).T.astype(np.float32))
y_train = torch.from_numpy(mu_c[:,np.newaxis].astype(np.float32))


model  = DirectNet_20(2, 1)
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-4)
loss_scale = 1000
n_epochs = 100000
for epoch in range(n_epochs):
	y_pred = -torch.square(model(x_train))
	loss = loss_fn(y_pred,y_train)*loss_scale

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	if epoch % 100 == 0:
		print("[{}/{}], loss: {}, time {}".format(epoch, n_epochs, np.round(loss.item(), 3),datetime.now()))
		torch.save(model, "visc.model")

y_pred_train = -model(x_train).detach().numpy().flatten()**2


yy = np.linspace(-2*np.pi, 2*np.pi, Ny)
plt.plot(yy, mu_c,  '--o', fillstyle="none", markevery = 1, markersize = 3, label="mu")
plt.plot(yy, y_pred_train,  '--o', fillstyle="none", markevery = 1, markersize = 3, label="NN(q, dq_dy)")

plt.xlim([-2,2])
plt.legend()
plt.xlabel("y")
plt.savefig("NN-fit.pdf")
# traced_fn = torch.jit.trace(model , (torch.rand(1, 1),))

# traced_fn = torch.jit.script(model)
#my_script_module = torch.jit.script

# omega = np.zeros(100, dtype="float32")

# omega = np.zeros(100)

# model(torch.reshape(torch.tensor(omega, dtype=torch.float32), (100,1))).detach().numpy()