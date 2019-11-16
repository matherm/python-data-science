import numpy as np
import matplotlib.pyplot as plt

def rbf(x,y):
    u = np.linalg.norm(x-y, axis=1)
    return np.exp(-u)

def sigmoid(x,y):
    return np.tanh(np.dot(x, y.T))

def relu(x,y):
    # Ch. 2.1 https://cseweb.ucsd.edu/~saul/papers/nips09_kernel.pdf
    z = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    dot = np.dot(x, y.T).flatten()
    phi = np.arccos(dot / z)
    coeff = np.asarray([1/np.pi, 1/2, 1/(2*np.pi), 0, 1/24*np.pi]) # (5,)
    feature = np.asarray([phi**0,phi**1, phi**2, phi**3, phi**4 ]) # (5, N, N)
    return np.dot(coeff, feature)

def polynomial(x,y):
    return np.dot(x, y.T)**2

N = 1000
x = np.linspace(-5,5,N)
y = np.linspace(-5,5,N)
samples = np.asarray(np.meshgrid(x,y)) # (2, N, N)
data = samples.transpose(1, 2, 0).copy().reshape(N*N,-1) # (N*N, 2)
assert data.shape == (N*N, 2)
origin = np.array([[0.,0.]]) + 1e-1

Ts = ["rbf", "sigmoid", "relu", "polynomial (quadratic)"]
Ks = [rbf(origin, data),
      sigmoid(origin, data),
      relu(origin, data),
      polynomial(origin, data)]

for K,T in zip(Ks,Ts):
    grid = np.abs(K.reshape(N,N))
    plt.title("Kernel absolute values in range [-5, 5] for " + T)
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig(T)


