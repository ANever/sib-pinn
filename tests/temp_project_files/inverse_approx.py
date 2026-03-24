import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
def basis(x):
    return(1, x**(1/2), x, x**2, x**3, x**4, np.exp(-x), np.exp(-2*x))

def inv_func(x_):
    return fsolve(lambda x:x*(2+np.exp(x)+np.exp(-x))-x_, 0, xtol=1e-8)[0]

def func(x):
    return x*(2+np.exp(x)+np.exp(-x))
    
#func = inv_func

n = len(basis(0))
N =10000
A = np.empty((N,n))
y0 = np.zeros(N)
X = np.linspace(0,4,N)
for i, x in enumerate(X):
#for i, x in enumerate(np.linspace(0,200,N)):
    #print(func(x))
    y0[i] = func(x)
    A[i] = basis(y0[i])
res = np.linalg.solve(np.dot(A.T,A), np.dot(A.T,X))
#res = np.linalg.lstsq(A, X)
print(res)

tab0 = np.zeros(N)
tab2 = np.zeros(N)
XX = np.linspace(0,200,N)
for i, y in enumerate(XX):
    tab2[i] = basis(y) @ res
    tab0[i] = inv_func(y)
#plt.plot(tab)
plt.plot(y0, X)
plt.plot(XX, tab2)
plt.plot(XX, tab0)
plt.show()