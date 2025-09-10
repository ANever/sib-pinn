"""
********************************************************************************
training
********************************************************************************
"""

import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt


f = lambda t, x: -x
a = 0  # left bound
b = 1  # right bound
N = 100  # number of points
t = np.linspace(a, b, N)
x_exact = np.vectorize(lambda t: np.exp(-t))(t)  # equation solution


beta = 12
gamma = 3


def f(t, x):
    s, i = x
    ds = -beta * s * i
    di = beta * s * i - gamma * i
    return np.array([ds, di])


x = np.zeros((len(t), 2))
x[0] = [0.9, 0.1]


def euler(t, u, f):
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]  # step
        u[i + 1] = u[i] + h * (f(t[i], u[i]))
    return u


# plt.plot(t, np.abs(euler(t,x,f)[:,0]))
plt.plot(t, np.abs(euler(t, x, f)[:, 1]))
# plt.show()

# plt.rcPariams.update({"text.usetex": True, "font.family": "Helvetica"})

names = ["1111", "111-10", "111-10", "g1111", "g111-10"]  # , "g11-10-5"]
labels = [
    "Baseline",
    "beta (1,1)",
    "beta (1,10)",
    "beta (5,10)",
    "gamma (1,1)",
    "gamma (1,10)",
    "gamma (5,10)",
]
for name in names:
    with open(name + "/I" + ".pickle", "rb") as handle:
        u_n = pickle.load(handle)
    u = np.sum(u_n[1].reshape((100, 100)), axis=0) / 100
    # print(u_n.shape)
    plt.plot(t, u)
plt.legend(labels)
plt.savefig("plotI.png")
