import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import imageio

relax_list_training = [0.01, 0.02, 0.04, 0.08]
relax_list_test = [0.16]
color_list = ['b', 'g', 'r', 'c', 'k']
int_pts_idx = slice(20, 364)
t_avg_idx = 600

w_training = []
w_test = []
closure_training = []
closure_test = []

for idx, relax in enumerate(relax_list_training):
    fdir = '../matlab/output/beta_1.0_Gamma_1.0_relax_' + str(relax)
    data_w = scipy.io.loadmat(fdir + '/data_w.mat')
    data_w = data_w['data_w'][0]
    data_closure = scipy.io.loadmat(fdir + '/data_closure.mat')
    data_closure = data_closure['data_closure'][0]
    data_w = np.mean(data_w[int_pts_idx,t_avg_idx:-1], 1)
    data_closure = np.mean(data_closure[int_pts_idx,t_avg_idx:-1], 1) / relax_list_training[idx]
    w_training.append(data_w)
    closure_training.append(data_closure)
w_training = np.array(w_training)
w_training = w_training.flatten().reshape((-1,1))
closure_training = np.array(closure_training)
closure_training = closure_training.flatten().reshape((-1,1))

for idx, relax in enumerate(relax_list_test):
    fdir = '../matlab/output/beta_1.0_Gamma_1.0_relax_' + str(relax)
    data_w = scipy.io.loadmat(fdir + '/data_w.mat')
    data_w = data_w['data_w'][0]
    data_closure = scipy.io.loadmat(fdir + '/data_closure.mat')
    data_closure = data_closure['data_closure'][0]
    data_w = np.mean(data_w[int_pts_idx,t_avg_idx:-1], 1)
    data_closure = np.mean(data_closure[int_pts_idx,t_avg_idx:-1], 1) / relax_list_test[idx]
    w_test.append(data_w)
    closure_test.append(data_closure)
w_test = np.array(w_test)
w_test = w_test.flatten().reshape((-1,1))
closure_test = np.array(closure_test)
closure_test = closure_test.flatten().reshape((-1,1))

torch.manual_seed(1) 

w_training = torch.from_numpy(w_training)
w_training = w_training.float()
closure_training = torch.from_numpy(closure_training)
closure_training = closure_training.float()
x, y = Variable(w_training), Variable(closure_training)

w_test = torch.from_numpy(w_test)
w_test = w_test.float()
closure_test = torch.from_numpy(closure_test)
closure_test = closure_test.float()
x_test, y_test = Variable(w_test), Variable(closure_test)

# another way to define a network
net = torch.nn.Sequential(
        torch.nn.Linear(1, 200),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(200, 100),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 1),
    )

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

BATCH_SIZE = 16 
EPOCH = 200

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, num_workers=2,)

my_images = []
fig, ax = plt.subplots(figsize=(16,10))

# start training
for epoch in range(EPOCH):
    print("Epoch " + str(epoch))
    for step, (batch_x, batch_y) in enumerate(loader): # for each training step
        
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        prediction = net(b_x)     # input x and predict based on x

        loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

        if step == 1:
            # plot and show learning process
            plt.cla()
            ax.set_xlabel(r'$\langle \omega \rangle$')
            ax.set_ylabel(r'$\langle \mathbf{v} \cdot \nabla \omega \rangle$')
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1., 1.)
            ax.scatter(b_x.data.numpy(), b_y.data.numpy(), color = "blue", alpha=0.2)
            ax.scatter(b_x.data.numpy(), prediction.data.numpy(), color='green', alpha=0.5)
            ax.text(0.6, -0.5, 'Epoch = %d' % epoch,
                    fontdict={'size': 20, 'color':  'red'})
            ax.text(0.6, -0.65, 'Loss = %.6f' % loss.data.numpy(),
                    fontdict={'size': 20, 'color':  'red'})

            # Used to return the plot as an image array 
            fig.canvas.draw()       # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            my_images.append(image)

# save images as a gif    
imageio.mimsave('./training_batch.gif', my_images, fps=5)

plt.figure()
plt.plot(x.data.numpy(), y.data.numpy(), 'o', color = "blue", alpha=0.2, label='training data')
prediction = net(x)     # input x and predict based on x
plt.plot(x.data.numpy(), prediction.data.numpy(), 'o', color='green', alpha=0.5, label='NN')
plt.xlabel(r'$\langle \omega \rangle$')
plt.ylabel(r'$\langle \mathbf{v} \cdot \nabla \omega \rangle$')
plt.legend()
plt.tight_layout()
plt.savefig('trained_model.pdf')
plt.close()

plt.figure()
plt.plot(x_test.data.numpy(), y_test.data.numpy(), 'o', color = "blue", alpha=0.2, label='test data')
prediction = net(x_test)     # input x and predict based on x
plt.plot(x_test.data.numpy(), prediction.data.numpy(), 'o', color='green', alpha=0.5, label='NN prediction')
plt.xlabel(r'$\langle \omega \rangle$')
plt.ylabel(r'$\langle \mathbf{v} \cdot \nabla \omega \rangle$')
plt.legend()
plt.tight_layout()
plt.savefig('test_model.pdf')
plt.close()
