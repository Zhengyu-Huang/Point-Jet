from Utility import gradient_first
import numpy as np
import scipy.io
import scipy.ndimage
import matplotlib.pyplot as plt
from NeuralNet import *


Ny = 256
L = 32
H = [1.0, 1.0]               # the rest depths of each layer
lam = 0.25
U = [1.0, -1.0]
dU = U[0] - U[1] 
F1 = 2.0/lam**2
F2 = 2.0/lam**2


# 'beta12rek0p32'
def preprocess_data(file_name, beta, lam, dU, L):
    
    F1 = 2.0/lam**2
    F2 = 2.0/lam**2
    beta1, beta2 = beta + F1*dU, beta - F2*dU


    pre_file  = '/central/groups/esm/zhaoyi/pyqg_run/2layer/' + file_name + '/'
    start, end, step = 3000000, 6000000, 20000
    u, v, q, psi = load_netcdf(pre_file, file_name, start, end, step)
    
    nt, nx, ny, nlayers = u.shape

    q_zonal_mean   = np.mean(q, axis = 1)
    psi_zonal_mean = np.mean(psi, axis = 1)
    dq_zonal_mean  = np.copy(q_zonal_mean)
    
    yy, dy = np.linspace(0, L - L/ny, ny), L/ny

    flux_zonal_mean = np.mean(v * q, axis = 1)
    for i in range(nt):
        for j in range(nlayers):
            dq_zonal_mean[i, :, j] = gradient_first(q_zonal_mean[i, :, j], dy)


    dpv_zonal_mean =  np.copy(dq_zonal_mean)
    dpv_zonal_mean[:,:, 0] += beta1
    dpv_zonal_mean[:,:, 1] += beta2    
    
    t_mean_steps = range(0,nt)
    flux_mean    = np.mean(flux_zonal_mean[t_mean_steps, :, :], axis = 0)
    flux_mean[:, 0] = np.mean(flux_mean[:, 0])
    dpv_mean     = np.mean(dpv_zonal_mean[t_mean_steps, :, :],  axis = 0)
    q_mean       = np.mean(q_zonal_mean[t_mean_steps, :, :],    axis = 0)

    
    mu_mean = -flux_mean / dpv_mean

    
    return mu_mean, dpv_mean



n_exp = 1
nx = ny = 256
n_feature = 1
x_train_top = np.zeros((nx*n_exp, n_feature))
x_train_bot = np.zeros((nx*n_exp, n_feature))
y_train_top = np.zeros((nx*n_exp, 1))
y_train_bot = np.zeros((nx*n_exp, 1))

file_names = ['beta12rek0p32'] 
betas = [12]
for i in range(n_feature):
    mu_mean, dpv_mean = preprocess_data(file_name, beta, lam, dU, L)
    x_train_top[i*nx:(i+1)*nx, 0] = dpv_mean[0, :]
    x_train_bot[i*nx:(i+1)*nx, 0] = dpv_mean[1, :]
    y_train_top[i*nx:(i+1)*nx, 0] = mu_mean[0, :]
    y_train_bot[i*nx:(i+1)*nx, 0] = mu_mean[1, :]
    



x_train = torch.from_numpy(x_train_top.astype(np.float32))
y_train = torch.from_numpy(y_train_top.astype(np.float32))



ind = 2
outd = 1 
layers = 2
width = 20

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
learning_rate = 0.001
epochs = 1000
step_size = 100
gamma = 0.5




model = FNN(ind, outd, layers, N_neurons) 


optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = torch.nn.MSELoss(reduction='sum')
y_normalizer.cuda()
t0 = default_timer()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:

        batch_size_ = x.shape[0]
        optimizer.zero_grad()
        out = model(x)

        loss = myloss(out , y)
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    
    scheduler.step()

    train_l2/= ntrain

    t2 = default_timer()
    print("Epoch : ", ep, " Epoch time : ", t2-t1, " Train L2 Loss : ", train_l2)

torch.save(model, "2-layer.model")

print("Total time is :", default_timer() - t0, "Total epoch is ", epochs)





# y_pred_train = -model(torch.from_numpy(np.stack((q_mean_abs, dq_dy_mean)).T.astype(np.float32))).detach().numpy().flatten()

# # plot data
# plt.figure()
# plt.plot(q_mean, dq_dy_mean, '--o', fillstyle="none", markevery = 1, markersize = 3, label="q")
# plt.xlabel("q")
# plt.ylabel("dq_dy")
# plt.show()



# plt.figure()
# # plt.plot(yy, q_mean, '--o', fillstyle="none", markevery = 1, markersize = 3, label="q")
# # plt.plot(yy, dq_dy_mean, '--o', fillstyle="none", markevery = 1, markersize = 3, label="dq_dy")

# plt.plot(yy, (closure_mean[chop_l:-chop_l]/dq_dy_mean), '--o', fillstyle="none", markevery = 1, markersize = 3, label="100*mu")
# plt.plot(yy, mu_c,  '--o', fillstyle="none", markevery = 1, markersize = 3, label="100*filtered mu")
# plt.plot(yy, y_pred_train,  '--o', fillstyle="none", markevery = 1, markersize = 3, label="100*NN(q, dq_dy)")

# # plt.xlim([-2,2])
# plt.legend()
# plt.xlabel("y")
# plt.show()
# plt.savefig("NN-fit.pdf")
# # traced_fn = torch.jit.trace(model , (torch.rand(1, 1),))

# # traced_fn = torch.jit.script(model)
# #my_script_module = torch.jit.script

# # omega = np.zeros(100, dtype="float32")

# # omega = np.zeros(100)

# # model(torch.reshape(torch.tensor(omega, dtype=torch.float32), (100,1))).detach().numpy()