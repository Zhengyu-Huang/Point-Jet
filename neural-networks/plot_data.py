import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb

relax_list = [0.01, 0.02, 0.04, 0.08, 0.16]
color_list = ['b', 'g', 'r', 'c', 'k']
int_pts_idx = slice(20, 364)
t_avg_idx = 600

plt.figure()
for idx, relax in enumerate(relax_list):
    fdir = '../matlab/output/beta_1.0_Gamma_1.0_relax_' + str(relax)
    data_w = scipy.io.loadmat(fdir + '/data_w.mat')
    data_w = data_w['data_w'][0]
    data_closure = scipy.io.loadmat(fdir + '/data_closure.mat')
    data_closure = data_closure['data_closure'][0]
    data_w = np.mean(data_w[int_pts_idx,t_avg_idx:-1], 1)
    data_closure = np.mean(data_closure[int_pts_idx,t_avg_idx:-1], 1)
    plt.plot(data_w, data_closure / relax_list[idx], 'o', color = color_list[idx], \
             label = 'relax = ' + str(relax_list[idx]))
plt.xlabel(r'$\langle \omega \rangle$')
plt.ylabel(r'$\langle \mathbf{v} \cdot \nabla \omega \rangle$')
plt.legend()
plt.tight_layout()
plt.savefig('closure_comparison.pdf')
plt.close()
