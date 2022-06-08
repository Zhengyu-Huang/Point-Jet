import numpy as np
import scipy
from scipy.linalg import block_diag
import multiprocessing
import pickle
import os
"""
UKI{FT<:AbstractFloat, IT<:Int}
Struct that is used in Unscented Kalman Inversion (UKI)
For solving the inverse problem 
    y = G(theta) + eta
    
"""
class UKI:
    def __init__(self, theta_names,
                 theta0_mean, theta0_mean_init, 
                 theta0_cov, theta0_cov_init, 
                 y,
                 Sigma_eta,  alpha_reg,  gamma,  update_freq, 
                 modified_uscented_transform = True):

        N_theta = theta0_mean.size
        N_y = y.size
        # ensemble size
        N_ens = 2*N_theta+1

    
        c_weights = np.zeros(N_theta+1)
        mean_weights = np.zeros(N_ens)
        cov_weights = np.zeros(N_ens)

        # todo parameters lam, alpha, beta

        beta = 2.0
        alpha = min(np.sqrt(4/(N_theta)), 1.0)
        lam = alpha**2*(N_theta) - N_theta


        c_weights[1:N_theta+1]  =  np.sqrt(N_theta + lam)
        mean_weights[0] = lam/(N_theta + lam)
        mean_weights[1:N_ens] = 1/(2*(N_theta + lam))
        cov_weights[0] = lam/(N_theta + lam) + 1 - alpha**2 + beta
        cov_weights[1:N_ens] = 1/(2*(N_theta + lam))

        if modified_uscented_transform:
            mean_weights[0] = 1.0
            mean_weights[1:N_ens] = 0.0


        

        theta_mean = []  # array of Array{FT, 2}'s
        theta_mean.append(theta0_mean_init) # insert parameters at end of array (in this case just 1st entry)
        theta_cov = [] # array of Array{FT, 2}'s
        theta_cov.append(theta0_cov_init) # insert parameters at end of array (in this case just 1st entry)

        y_pred = []  # array of Array{FT, 2}'s
    

        Sigma_omega = (gamma + 1 - alpha_reg**2)*theta0_cov
        Sigma_nu = (gamma + 1)/gamma*Sigma_eta

        r = theta0_mean
        
        iter = 0


        "vector of parameter names (never used)"
        self.theta_names = theta_names
        "a vector of arrays of size N_ensemble x N_parameters containing the mean of the parameters (in each uki iteration a new array of mean is added)"
        self.theta_mean = theta_mean
        "a vector of arrays of size N_ensemble x (N_parameters x N_parameters) containing the covariance of the parameters (in each uki iteration a new array of cov is added)"
        self.theta_cov = theta_cov
        "a vector of arrays of size N_ensemble x N_y containing the predicted observation (in each uki iteration a new array of predicted observation is added)"
        self.y_pred = y_pred
        "vector of observations (length: N_y)"
        self.y = y
        "covariance of the observational noise"
        self.Sigma_eta = Sigma_eta
        "number ensemble size (2N_theta - 1)"
        self.N_ens = N_ens
        "size of theta"
        self.N_theta = N_theta
        "size of y"
        self.N_y = N_y
        "weights in UKI"
        self.c_weights = c_weights
        self.mean_weights = mean_weights
        self.cov_weights = cov_weights
        "covariance of the artificial evolution error"
        self.Sigma_omega = Sigma_omega
        "covariance of the artificial observation error"
        self.Sigma_nu = Sigma_nu
        "regularization parameter"
        self.alpha_reg = alpha_reg 
        "gamma for the error covariance matrices"
        self.gamma = gamma
        "regularization vector"
        self.r = r
        "update frequency"
        self.update_freq = update_freq
        "current iteration number"
        self.iter = iter




"""
UKI Constructor 
parameter_names::Array{String,1} : parameter name vector
theta0_mean::Array{FT} : prior mean
theta0_cov::Array{FT, 2} : prior covariance
g_t::Array{FT,1} : observation 
obs_cov::Array{FT, 2} : observation covariance
alpha_reg::FT : regularization parameter toward theta0 (0 < alpha_reg <= 1), default should be 1, without regulariazion
"""



"""
construct_sigma_ensemble
Construct the sigma ensemble, based on the mean x_mean, and covariance x_cov
"""
def construct_sigma_ensemble(uki, x_mean, x_cov):
    N_ens = uki.N_ens
    N_x = x_mean.size

    c_weights = uki.c_weights

    # chol_xx_cov = np.linalg.cholesky((x_cov + x_cov.T)/2.0)  #cholesky(Hermitian(x_cov)).L
    
    
    u, s, _ = np.linalg.svd(x_cov, full_matrices=True)
    chol_xx_cov = u * np.sqrt(s)
    

    x = np.zeros((2*N_x+1, N_x))
    x[0, :] = x_mean
    for i in range(1, N_x+1):
        x[i,     :] = x_mean + c_weights[i]*chol_xx_cov[:,i-1]
        x[i+N_x, :] = x_mean - c_weights[i]*chol_xx_cov[:,i-1]
    
    return x



"""
construct_mean x_mean from ensemble x
"""
def construct_mean(uki, x):
    N_ens, N_x = x.shape

    # @assert(uki.N_ens == N_ens)

    x_mean = np.zeros( N_x)
    mean_weights = uki.mean_weights
 
    for i in range(N_ens):
        x_mean += mean_weights[i]*x[i,:]


    return x_mean


# """
# construct_cov xx_cov from ensemble x and mean x_mean
# """
# def construct_cov(uki, x, x_mean):
#     N_ens, N_x = uki.N_ens, x_mean.shape[0]
    
#     cov_weights = uki.cov_weights

#     xx_cov = np.zeros((N_x, N_x))

#     for i in range(N_ens):
#         xx_cov += cov_weights[i]*np.outer(x[i,:] - x_mean, x[i,:] - x_mean)

#     return xx_cov


"""
construct_cov xy_cov from ensemble x and mean x_mean, ensemble y and mean y_mean
"""
def construct_cov(uki, x, x_mean, y, y_mean):
    N_ens, N_x, N_y = uki.N_ens, x_mean.shape[0], y_mean.shape[0]
    
    cov_weights = uki.cov_weights

    xy_cov = np.zeros((N_x, N_y))

    for i in range(N_ens):
        xy_cov += cov_weights[i]*np.outer(x[i,:] - x_mean , y[i,:] - y_mean)
    return xy_cov


# def reset_step(uki, gamma):
#     uki.gamma = gamma
    
#     uki.theta_mean.pop()
#     "a vector of arrays of size N_ensemble x (N_parameters x N_parameters) containing the covariance of the parameters (in each uki iteration a new array of cov is added)"
#     uki.theta_cov.pop()
#     "a vector of arrays of size N_ensemble x N_y containing the predicted observation (in each uki iteration a new array of predicted observation is added)"
#     uki.y_pred.pop()
#     "current iteration number"
#     uki.iter -= 1
    
    

# """
# update uki struct
# ens_func: The function g = G(theta)
# define the function as 
#     ens_func(theta_ens) = MyG(phys_params, theta_ens, other_params)
# use G(theta_mean) instead of FG(theta)
# """
# def update_ensemble(uki, ens_func):
    
#     uki.iter += 1
#     # update evolution covariance matrix
#     if uki.update_freq > 0 and uki.iter%uki.update_freq == 0:
#         uki.Sigma_omega = (2 - uki.alpha_reg**2) * uki.theta_cov[-1]


#     theta_mean  = uki.theta_mean[-1]
#     theta_cov = uki.theta_cov[-1]
#     y = uki.y

#     alpha_reg = uki.alpha_reg
#     r = uki.r
#     Sigma_omega = uki.Sigma_omega
#     Sigma_nu = uki.Sigma_nu

#     N_theta, N_y, N_ens = uki.N_theta, uki.N_y, uki.N_ens
#     ############# Prediction step:

#     theta_p_mean  = alpha_reg*theta_mean + (1-alpha_reg)*r
#     theta_p_cov = alpha_reg**2*theta_cov + Sigma_omega
    


#     ############ Generate sigma points
#     theta_p = construct_sigma_ensemble(uki, theta_p_mean, theta_p_cov)
#     # play the role of symmetrizing the covariance matrix
#     theta_p_cov = construct_cov(uki, theta_p, theta_p_mean)

#     ###########  Analysis step
#     g = np.zeros((N_ens, N_y))
#     g[:,:] = ens_func(theta_p)

#     g_mean = construct_mean(uki, g)
#     gg_cov = construct_cov(uki, g, g_mean) + Sigma_nu
#     thetag_cov = construct_cov(uki, theta_p, theta_p_mean, g, g_mean)

#     tmp = np.linalg.solve(gg_cov.T, thetag_cov.T).T 

#     theta_mean =  theta_p_mean + tmp*(y - g_mean)

#     theta_cov =  theta_p_cov - tmp*thetag_cov.T 


#     ########### Save resutls
#     uki.y_pred.append(g_mean) # N_ens x N_data
#     uki.theta_mean.append(theta_mean) # N_ens x N_params
#     uki.theta_cov.append(theta_cov) # N_ens x N_data


def ensemble(s_param, theta_ens, forward, parallel_flag = True):
    
    N_ens,  N_theta = theta_ens.shape
    N_y = s_param.N_y
    g_ens = np.zeros((N_ens,  N_y))
    
    
    if parallel_flag == True:
        pool = multiprocessing.Pool(min(multiprocessing.cpu_count(), N_ens))
        results = []
        for i in range(N_ens):
            results.append(pool.apply_async(forward, (s_param, theta_ens[i,:], )))
        for (i, result) in enumerate(results):
            g_ens[i, :] = result.get()
        pool.close()
    
    
    else:
        for i in range(N_ens):
            theta = theta_ens[i, :]
            g_ens[i, :] = forward(s_param, theta)

    return g_ens


    
    
def update_prediction(uki):
    
    # update evolution covariance matrix
    uki.iter += 1
    if uki.update_freq > 0 and (uki.iter + 1)%uki.update_freq == 0:
        uki.Sigma_omega = (uki.gamma + 1 - uki.alpha_reg**2) * uki.theta_cov[-1]
        uki.Sigma_nu = (uki.gamma + 1)/uki.gamma * uki.Sigma_eta


    theta_mean  = uki.theta_mean[-1]
    theta_cov = uki.theta_cov[-1]
    y = uki.y

    alpha_reg = uki.alpha_reg
    r = uki.r
    Sigma_omega = uki.Sigma_omega
    Sigma_nu = uki.Sigma_nu

    N_theta, N_y, N_ens = uki.N_theta, uki.N_y, uki.N_ens
    ############# Prediction step:
    

    theta_p_mean  = alpha_reg*theta_mean + (1-alpha_reg)*r
    theta_p_cov = alpha_reg**2*theta_cov + Sigma_omega
    
    

    ############ Generate sigma points
    theta_p = construct_sigma_ensemble(uki, theta_p_mean, theta_p_cov)
    

    return theta_p

def update_analysis(uki, theta_p, g):    
    
    Sigma_nu, y = uki.Sigma_nu, uki.y
    
    theta_p_mean = construct_mean(uki, theta_p)
    # play the role of symmetrizing the covariance matrix
    theta_p_cov = construct_cov(uki, theta_p, theta_p_mean, theta_p, theta_p_mean)
    

    ###########  Analysis step

    g_mean = construct_mean(uki, g)
    gg_cov = construct_cov(uki, g, g_mean, g, g_mean) + Sigma_nu
    thetag_cov = construct_cov(uki, theta_p, theta_p_mean, g, g_mean)

    tmp = np.linalg.solve(gg_cov.T, thetag_cov.T).T 

    
    theta_mean =  theta_p_mean + np.matmul(tmp, (y - g_mean))
    theta_cov =  theta_p_cov - np.matmul(tmp, thetag_cov.T)
    
    

#     ########### Save resutls
    uki.y_pred.append(g_mean) # N_ens x N_data
    uki.theta_mean.append(theta_mean) # N_ens x N_params
    uki.theta_cov.append(theta_cov) # N_ens x N_data
    
    
    return theta_mean, theta_cov



def UKI_Run(s_param, forward, 
    theta0_mean, theta0_mean_init, 
    theta0_cov,  theta0_cov_init,
    y, Sigma_eta,
    alpha_reg,
    gamma,
    update_freq,
    N_iter,
    save_folder = "data",
    modified_uscented_transform = True,
    theta_basis = None):
    
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    theta_names = s_param.theta_names
    
    
    ukiobj = UKI(theta_names ,
    theta0_mean, theta0_mean_init,
    theta0_cov, theta0_cov_init,
    y,
    Sigma_eta,
    alpha_reg,
    gamma,
    update_freq,
    modified_uscented_transform = modified_uscented_transform)
    
    if theta_basis == None:
        ens_func = lambda theta_ens : ensemble(s_param, theta_ens, forward)
    else: 
        # theta_ens is N_ens × N_theta
        # theta_basis is N_theta × N_theta_high
        ens_func = lambda theta_ens : ensemble(s_param, np.matmul(theta_ens,theta_basis), forward)
    
    
    
    opt_errors = []
    
    i = 0
    pickle.dump(ukiobj, open(save_folder+"/ukiobj-" + str(i) + ".dat", "wb" ) )
    
    decrease_step = 0
    while i < N_iter:
        
        theta_p = update_prediction(ukiobj) 
        g = ens_func(theta_p)
        update_analysis(ukiobj, theta_p, g) 
        
        
        y_pred = ens_func( np.reshape(ukiobj.theta_mean[-1], (1, len(ukiobj.theta_mean[-1])))).flatten()
        opt_error = 0.5*np.dot((y_pred - ukiobj.y) , np.linalg.solve(ukiobj.Sigma_eta, (y_pred - ukiobj.y)))
        
        if i > 0 and opt_error > opt_errors[-1]:
            ukiobj.gamma = np.maximum(0.0625, ukiobj.gamma/2.0)
            decrease_step = 0
        else:
            decrease_step += 1
            if decrease_step >= 5:
                ukiobj.gamma = np.minimum(2.0, 2.0*ukiobj.gamma)
                decrease_step = 0
            
        opt_errors.append(opt_error)
        
            
        print("ukiobj.gamma : ", ukiobj.gamma)
        print( "optimization error at iter ", i, " = ", opt_errors[i] )
        
        N_theta, N_y = len(theta0_mean), len(y)
        print("Parameters are: ", ukiobj.theta_mean[-1])
        print("data-misfit : ", 0.5*np.dot((y_pred[0:N_y-N_theta] - ukiobj.y[0:N_y-N_theta]) , np.linalg.solve(ukiobj.Sigma_eta[0:N_y-N_theta,0:N_y-N_theta], (y_pred[0:N_y-N_theta] - ukiobj.y[0:N_y-N_theta]))),  
              "reg : ", 0.5*np.dot((y_pred[-N_theta:] - ukiobj.y[-N_theta:]) , np.linalg.solve(ukiobj.Sigma_eta[-N_theta:,-N_theta:], (y_pred[-N_theta:] - ukiobj.y[-N_theta:]))))
        print( "Frobenius norm of the covariance at iter ", i, " = ", np.linalg.norm(ukiobj.theta_cov[i]) ) 
        i += 1
        
        pickle.dump(ukiobj, open(save_folder+"/ukiobj-" + str(i) + ".dat", "wb" ) )
        
        
        
    return ukiobj
    

# def UKI_Run(s_param, forward, 
#     theta0_mean, theta0_cov,
#     y, Sigma_eta,
#     alpha_reg,
#     gamma,
#     update_freq,
#     N_iter,
#     modified_uscented_transform = True,
#     theta_basis = None):
    
#     theta_names = s_param.theta_names
    
    
#     ukiobj = UKI(theta_names ,
#     theta0_mean, 
#     theta0_cov,
#     y,
#     Sigma_eta,
#     alpha_reg,
#     gamma,
#     update_freq,
#     modified_uscented_transform = modified_uscented_transform)
    
#     if theta_basis == None:
#         ens_func = lambda theta_ens : ensemble(s_param, theta_ens, forward)
#     else: 
#         # theta_ens is N_ens × N_theta
#         # theta_basis is N_theta × N_theta_high
#         ens_func = lambda theta_ens : ensemble(s_param, np.matmul(theta_ens,theta_basis), forward)
    
    
    
#     opt_errors = []
    
#     i = 0
#     while i < N_iter:
        
#         theta_p = update_prediction(ukiobj) 
#         g = ens_func(theta_p)
        
#         y_pred = construct_mean(ukiobj, g)
        
        
#         opt_error = 0.5*np.dot(y_pred - ukiobj.y , np.linalg.solve(ukiobj.Sigma_eta, y_pred - ukiobj.y))
#         opt_errors.append(opt_error)
#         if i > 0 and opt_errors[i] > opt_errors[i-1]:
#             ukiobj.gamma = ukiobj.gamma/2.0
            
# #             theta_p = update_prediction(ukiobj) 
# #             g = ens_func(theta_p)
#         else:
#             ukiobj.gamma = np.minimum(1.0, 2.0*ukiobj.gamma)
          
        
#         update_analysis(ukiobj, theta_p, g) 
        
          

#         print("ukiobj.gamma : ", ukiobj.gamma)
#         print("len(ukiobj.opt_error) : ", i, len(opt_errors))
#         print( "optimization error at iter ", i, " = ", opt_errors[i] )
#         print("len(ukiobj.theta_cov) : ", i, len(ukiobj.theta_cov))
#         print( "Frobenius norm of the covariance at iter ", i, " = ", np.linalg.norm(ukiobj.theta_cov[i]) ) 

#         i += 1
        
#     return ukiobj

if __name__ == "__main__":

    class Param():
        def __init__(self, theta_names, N_theta,  N_y):
            self.theta_names = theta_names
            self.N_theta = N_theta
            self.N_y = N_y

    problem_type = "over-determined"
    if problem_type == "under-determined":
        # under-determined case
        theta_ref = np.array([0.6, 1.2])
        G = np.array([1.0, 2.0])
        y = np.array([3.0])
    elif problem_type == "over-determined":
        # over-determined case
        theta_ref = np.array([1/3, 8.5/6])
        G = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([3.0, 7.0, 10.0])
    else:
        print("Problem type : ", problem_type, " has not implemented!")
    
    Sigma_eta = np.zeros((len(y), len(y)))
    np.fill_diagonal(Sigma_eta, 0.1**2)
    

    
    def linear_aug(s_param, theta, G = G):        
        return np.hstack((np.matmul(G, theta), theta))
    
    N_theta = len(theta_ref)
    s_param = Param(["theta1, theta2"], N_theta, len(y) + N_theta)
    theta0_mean = np.zeros(N_theta)
    
    theta0_cov = np.zeros((N_theta, N_theta))
    np.fill_diagonal(theta0_cov, 1.0**2)  
    
    y_aug = np.hstack((y, theta0_mean))
    Sigma_eta_aug = block_diag(Sigma_eta, theta0_cov)
    Sigma_post = np.linalg.inv(np.matmul(G.T, np.linalg.solve(Sigma_eta, G)) + np.linalg.inv(theta0_cov))
    theta_post = theta0_mean + np.matmul(Sigma_post, (np.matmul(G.T, np.linalg.solve(Sigma_eta, (y - np.matmul(G,theta0_mean))))))
            

    alpha_reg = 1.0
    update_freq = 1
    N_iter = 30
    uki_obj = UKI_Run(s_param, linear_aug, 
        theta0_mean, theta0_cov,
        y_aug, Sigma_eta_aug,
        alpha_reg,
        update_freq,
        N_iter)





    uki_errors    = np.zeros((N_iter+1, 2))
    
    for i in range(N_iter+1):
        
        uki_errors[i, 0] = np.linalg.norm(uki_obj.theta_mean[i] - theta_post)/np.linalg.norm(theta_post)
        uki_errors[i, 1] = np.linalg.norm(uki_obj.theta_cov[i] - Sigma_post)/np.linalg.norm(Sigma_post)
    
    import  matplotlib.pyplot as plt
    ites = np.arange(0, N_iter+1)    
    fig, ax = plt.subplots(nrows = 1, ncols=2, sharex=False, sharey=False, figsize=(15,6))
    ax[0].semilogy(ites, uki_errors[:, 0],   "-.x", color = "C0", fillstyle="none", label="UKI")
    ax[0].set_xlabel("Iterations")
    ax[0].set_ylabel("Rel. mean error")
    ax[0].grid("on")
    ax[1].semilogy(ites, uki_errors[:, 1],   "-.x", color = "C0", fillstyle="none", label="UKI")
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("Rel. covariance error")
    ax[1].grid("on")
    ax[1].legend(bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    plt.show()
       
# function plot_param_iter(ukiobj::UKI{FT, IT}, theta_ref::Array{FT,1}, theta_ref_names::Array{String}) where {FT<:AbstractFloat, IT<:Int}
    
#     theta_mean = ukiobj.theta_mean
#     theta_cov = ukiobj.theta_cov
    
#     N_iter = length(theta_mean) - 1
#     ites = Array(LinRange(1, N_iter+1, N_iter+1))
    
#     theta_mean_arr = abs.(hcat(theta_mean...))
    
    
#     N_theta = length(theta_ref)
#     theta_std_arr = zeros(Float64, (N_theta, N_iter+1))
#     for i = 1:N_iter+1
#         for j = 1:N_theta
#             theta_std_arr[j, i] = np.sqrt(theta_cov[i][j,j])
#         end
#     end
    
#     for i = 1:N_theta
#         errorbar(ites, theta_mean_arr[i,:], yerr=3.0*theta_std_arr[i,:], fmt="--o",fillstyle="none", label=theta_ref_names[i])   
#         plot(ites, fill(theta_ref[i], N_iter+1), "--", color="gray")
#     end
    
#     xlabel("Iterations")
#     legend()
#     tight_layout()
# end


# function plot_opt_errors(ukiobj::UKI{FT, IT}, 
#     theta_ref::Union{Array{FT,1}, Nothing} = nothing, 
#     transform_func::Union{Function, Nothing} = nothing) where {FT<:AbstractFloat, IT<:Int}
    
#     theta_mean = ukiobj.theta_mean
#     theta_cov = ukiobj.theta_cov
#     y_pred = ukiobj.y_pred
#     Sigma_eta = ukiobj.Sigma_eta
#     y = ukiobj.y

#     N_iter = length(theta_mean) - 1
    
#     ites = Array(LinRange(1, N_iter, N_iter))
#     N_subfigs = (theta_ref === nothing ? 2 : 3)

#     errors = zeros(Float64, N_subfigs, N_iter)
#     fig, ax = PyPlot.subplots(ncols=N_subfigs, figsize=(N_subfigs*6,6))
#     for i = 1:N_iter
#         errors[N_subfigs - 1, i] = 0.5*(y - y_pred[i])'*(Sigma_eta\(y - y_pred[i]))
#         errors[N_subfigs, i]     = norm(theta_cov[i])
        
#         if N_subfigs == 3
#             errors[1, i] = norm(theta_ref - (transform_func === nothing ? theta_mean[i] : transform_func(theta_mean[i])))/norm(theta_ref)
#         end
        
#     end

#     markevery = max(div(N_iter, 10), 1)
#     ax[N_subfigs - 1].plot(ites, errors[N_subfigs - 1, :], linestyle="--", marker="o", fillstyle="none", markevery=markevery)
#     ax[N_subfigs - 1].set_xlabel("Iterations")
#     ax[N_subfigs - 1].set_ylabel("Optimization error")
#     ax[N_subfigs - 1].grid()
    
#     ax[N_subfigs].plot(ites, errors[N_subfigs, :], linestyle="--", marker="o", fillstyle="none", markevery=markevery)
#     ax[N_subfigs].set_xlabel("Iterations")
#     ax[N_subfigs].set_ylabel("Frobenius norm of the covariance")
#     ax[N_subfigs].grid()
#     if N_subfigs == 3
#         ax[1].set_xlabel("Iterations")
#         ax[1].plot(ites, errors[1, :], linestyle="--", marker="o", fillstyle="none", markevery=markevery)
#         ax[1].set_ylabel("L₂ norm error")
#         ax[1].grid()
#     end
    
# end