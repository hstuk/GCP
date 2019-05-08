import math

import numpy as np
import theano.tensor as T

import special_functions

pi = float(math.pi)

def kl_div_ng_ng(p_alpha, p_beta, p_nu, p_mu, q_alpha, q_beta,q_nu, q_mu):
        kl_dist = 1.0/2.0*p_alpha/p_beta*(q_mu-p_mu)**2.0*q_nu
        kl_dist = kl_dist+1.0/2.0*q_nu/p_nu
        kl_dist = kl_dist-1.0/2.0*T.log(q_nu/p_nu)
        kl_dist = kl_dist - 1.0/2.0 + q_alpha*T.log(p_beta/q_beta) - T.log(T.gamma(p_alpha)/T.gamma(q_alpha))
        kl_dist = kl_dist + (p_alpha - q_alpha)*special_functions.psi(p_alpha)-(p_beta - q_beta)*p_alpha/p_beta
        return kl_dist


def kl_div_ng_ng_with_real_psi(p_alpha, p_beta, p_nu, p_mu, q_alpha, q_beta,q_nu, q_mu):
        kl_dist = 1.0/2.0*p_alpha/p_beta*(q_mu-p_mu)**2.0*q_nu
        kl_dist = kl_dist+1.0/2.0*q_nu/p_nu
        kl_dist = kl_dist-1.0/2.0*T.log(q_nu/p_nu)
        kl_dist = kl_dist - 1.0/2.0 + q_alpha*T.log(p_beta/q_beta) - T.log(T.gamma(p_alpha)/T.gamma(q_alpha))
        kl_dist = kl_dist + (p_alpha - q_alpha)*T.psi(p_alpha)-(p_beta - q_beta)*p_alpha/p_beta
        return kl_dist
        
        

def kl_div_dir_dir(p_a0, q_a0):
    a0 = T.sum(p_a0,axis=1)
    b0 = T.sum(q_a0,axis=1)
    #assume a0 = b0+1, since there was only one observation added
    kl_dist = T.log(T.gamma(a0)/T.gamma(b0))+T.sum(T.log(T.gamma(q_a0)/T.gamma(p_a0)),axis=1)
   # kl_dist = T.log(b0)-T.log(T.sum((p_a0-q_a0)*q_a0))
    kl_dist = kl_dist + T.sum((p_a0-q_a0)*(special_functions.psi(p_a0)-special_functions.psi(a0).reshape((a0.shape[0],1))),axis=1)
#    kl_dist = kl_dist + T.sum((p_a0-q_a0)*(special_functions.psi(p_a0)-special_functions.psi(a0)),axis=1)
    return kl_dist

def posterior_cat_dir(p_a0, observations):
    return p_a0 + 1.0*observations

def predictive_cat_dir(p_a0):
    return p_a0/T.sum(p_a0)

def posterior_nd_ng(p_alpha, p_beta, p_nu, p_mu, observations):
       n_mu = (p_nu*p_mu + observations)/(p_nu+1.0)
       n_nu = p_nu + 1.0
       n_alpha = p_alpha + 1.0/2.0
       n_beta = p_beta+p_nu/(p_nu+1.0)*(observations-p_mu)**2.0/2.0
       return n_alpha, n_beta, n_nu, n_mu

def predictive_nd_ng(p_alpha, p_beta, p_nu, p_mu):
        return p_mu, 2.0*p_alpha, p_beta*(p_nu+1.0)/(p_nu*p_alpha)

def st_d_logp(x,mu,nu,sigma2):
    x_p = (x-mu)/T.sqrt(sigma2)
    prob = T.log(T.gamma((nu+1.0)/2.0)/(T.gamma(nu/2.0)*T.sqrt(pi*nu*sigma2))*T.power(1.0+x_p**2/nu,-(nu+1)/2.0))
    return prob
    
def norm_d_logp(x,alpha,beta,nu,mu):
        
#    mu, nu2, sigma2 = predictive_nd_ng(alpha,beta,nu,mu)
#    mu, var = st_d_mean_var(mu,nu2,sigma2)    
        
    sigma2 = beta/alpha
    
    x_p2 = (x-mu)**2/(2*sigma2)
   
    prob = T.log(  T.sqrt(2*pi*sigma2)**(-1) * T.exp(-x_p2)  )
    return prob
    



def st_d_mean_var(mu, nu, sigma):
#    if(nu <= 2.0):
 #       return mu, 100000
  #  else:
    #nu[nu <= 2.0] = 2.00001
    return mu, sigma*nu/((nu-2.0))

def dir_mean_var(p_a0):
    a0 = np.sum(p_a0)
    return p_a0/a0, p_a0*(a0-p_a0)/(a0**2*((a0+1)))

def ng_mean_var(alpha, beta, nu, mu):
    #alpha[alpha <= 1.0] = 1.00005
 #       alpha = 1.005
    return mu, (beta/((alpha-1)*nu)), alpha/beta, alpha/beta**2
