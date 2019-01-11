# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt


def kernel(a, b, param):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)

n = 50
Xtest = np.linspace(-5, 5, n).reshape(-1,1)
K_ss = kernel(Xtest, Xtest, param = 0.1)
L = np.linalg.cholesky(K_ss + 1e-15*np.eye(n))

def sample_prior(Xtest, L, iter=5):
    for i in range(iter):    
        f_prior = np.dot(L, np.random.normal(size=(n,1)))
        plt.plot(Xtest, f_prior)
    plt.axis([-5,5,-3,3])
    plt.title("Gussian Process Prior")
    plt.show()

sample_prior(Xtest, L)

n = 50
Xtrain = np.linspace(-5, 5, n).reshape(-1,1)
ytrain = np.sin(Xtrain)
K_ss = kernel(Xtrain, Xtrain, param = 0.1)
L = np.linalg.cholesky(K_ss + 1e-15*np.eye(n))


def sample_post(Xtrain, Xtest, L):
    # Compute the mean at our test points.
    K_s = kernel(Xtrain, Xtest, param = 0.1)
    Lk = np.linalg.solve(L, K_s)
    mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((n,))
    
    # Compute the standard deviation so we can plot it
    s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
    stdv = np.sqrt(s2)
    # Draw samples from the posterior at our test points.
    L = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
    f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,3)))
    
    plt.plot(Xtrain, ytrain, 'bs', ms=8)
    plt.plot(Xtest, f_post)
    plt.gca().fill_between(Xtest.flat, mu-2*stdv, mu+2*stdv, color="#dddddd")
    plt.plot(Xtest, mu, 'r--', lw=2)
    plt.axis([-5, 5, -3, 3])
    plt.title('Three samples from the GP posterior')
    plt.show()
    
sample_post(Xtrain, Xtest, L)
