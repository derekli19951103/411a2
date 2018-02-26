from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import pickle

import os
from scipy.io import loadmat
from timeit import default_timer as timer

M = loadmat("mnist_all.mat")


def get_label(name):
    num = int(name[-1])
    l = [0] * 10
    l[num] = 1
    return l


def generate_x_y(setname):
    if setname == 'train':
        x = np.zeros((60000, 784))
    if setname == 'test':
        x = np.zeros((10000, 784))
    y = []
    i = 0
    for name, sets in M.items():
        if setname in name:
            for pic in sets:
                y.append(get_label(name))
                x[i] = pic.flatten() / 255.
                i += 1
    x = x.T
    x = vstack((ones((1, x.shape[1])), x))
    y = array(y).T
    return x, y


def softmax(y):
    return exp(y) / tile(sum(exp(y), 0), (len(y), 1))


def forward(x, w):
    L0 = dot(w.T, x)
    output = softmax(L0)
    return output


def NLL(p, y):
    return -sum(y * log(p))


def f(w, x, y):
    return NLL(forward(x, w), y)


def df(w, x, y):
    output = forward(x, w)
    return dot(x, (output - y).T)


def shuffle_input(x, y):
    combo = vstack((x, y))
    combo = combo.T
    np.random.shuffle(combo)
    combo = combo.T
    x = combo[:785]
    y = combo[785:]
    return x, y


def next_batch(x, y, batchSize):
    for i in np.arange(0, x.shape[1], batchSize):
        if (i + batchSize) < x.shape[1]:
            yield (x[:, [i, i + batchSize]], y[:, [i, i + batchSize]])
        else:
            yield (x[:, [i, x.shape[1] - 1]], y[:, [i, y.shape[1] - 1]])


def sgd(x, y, w, df, alpha, epoch):
    it = 0
    while it < epoch:
        x, y = shuffle_input(x, y)
        for (batchX, batchY) in next_batch(x, y, 128):
            w -= alpha * df(w, batchX, batchY)
        it += 1
    return w


def step_sgd(x, y, w, df, alpha, epoch):
    it = 0
    weights = []
    while it < 500:
        x, y = shuffle_input(x, y)
        for (batchX, batchY) in next_batch(x, y, 128):
            w -= alpha * df(w, batchX, batchY)
        it += 1
        if 500%epoch==0:
            weights.append((w[392 + 9][0], w[393 + 14][0]))
    return weights


def update_every_weight(x, y, w, wi, wj, alpha, output):
    error = output[wj] - y[wj]
    w[wi, wj] -= alpha * np.sum(error * x[wi])


def gradient_descend(x, y, w, alpha, epoch):
    it = 0
    while it < epoch:
        x, y = shuffle_input(x, y)
        for (batchX, batchY) in next_batch(x, y, 128):
            for wi in range(w.shape[0]):
                for wj in range(w.shape[1]):
                    output = forward(batchX, w)
                    update_every_weight(batchX, batchY, w, wi, wj, alpha, output)
        it += 1
    return w


def step_sgd_momentum(x, y, w, df, alpha, epoch):
    it = 0
    velocity = zeros_like(w)
    weights = []
    while it < 500:
        x, y = shuffle_input(x, y)
        for (batchX, batchY) in next_batch(x, y, 128):
            velocity = 0.99 * velocity + alpha * df(w, batchX, batchY)
            w -= velocity
        it += 1
        if 500%epoch==0:
            weights.append((w[392 + 9][0], w[393 + 14][0]))
    return weights


def sgd_momentum(x, y, w, df, alpha, epoch):
    it = 0
    velocity = zeros_like(w)
    while it < epoch:
        x, y = shuffle_input(x, y)
        for (batchX, batchY) in next_batch(x, y, 128):
            velocity = 0.99 * velocity + alpha * df(w, batchX, batchY)
            w -= velocity
        it += 1
    return w


def train_validating(w):
    x, y = generate_x_y('train')
    output = forward(x, w)
    a = output.T
    b = -1.0 * np.ones_like(a)
    b[np.arange(len(a)), a.argmax(1)] = 1
    output = b.T
    return np.count_nonzero(np.equal(output, y)) / (x.shape[1] * 1.0)


def testing(w):
    x, y = generate_x_y('test')
    output = forward(x, w)
    a = output.T
    b = -1.0 * np.ones_like(a)
    b[np.arange(len(a)), a.argmax(1)] = 1
    output = b.T
    return np.count_nonzero(np.equal(output, y)) / (x.shape[1] * 1.0)


def get_loss(W1, W2, w, x, y):
    w[392 + 9][0] = W1
    w[393 + 14][0] = W2
    cost = f(w, x, y)
    return cost


def get_traj(w, x, y, steps, momentum=False):
    w[392 + 9][0] = 7
    w[393 + 14][0] = 7
    if not momentum:
        traj = step_sgd(x, y, w, df, 0.06, steps)
    else:
        traj = step_sgd_momentum(x, y, w, df, 0.06 * (1 - 0.99), steps)
    return traj


def get_traj_1(w, x, y, steps, momentum=False):
    w[392 + 9][0] = 7
    w[393 + 14][0] = -2
    if not momentum:
        traj = step_sgd(x, y, w, df, 0.06, steps)
    else:
        traj = step_sgd_momentum(x, y, w, df, 0.06 * (1 - 0.99), steps)
    return traj


def finite_difference(x, y, theta, h):
    origin_t = f(theta, x, y)
    theta += np.full((theta.shape[0], theta.shape[1]), h)
    after_t = f(theta, x, y)
    finite_diff = (after_t - origin_t) / h
    total_error = sum(finite_diff - df(theta, x, y))
    return abs(total_error) / (785 * 10 * 1.0)


if __name__ == "__main__":
    fig, ax = plt.subplots(10, 10)
    for i in range(10):
        for j in range(10):
            ax[i, j].imshow(M["train" + str(i)][j].reshape((28, 28)), cmap=cm.gray)
            ax[i,j].axis('off')
    fig.suptitle('Digits')
    plt.savefig("part1.png")
    plt.gca().clear()
    print("part1.png")
    np.random.seed(1)
    W0 = np.random.randn(785, 10) / sqrt(785)
    x, y = generate_x_y('train')

    print('finite_difference:',finite_difference(x, y, W0, 1e-5))

    """sgd"""
    iteration = range(0, 2000, 100)
    success_t = []
    success_te = []
    for i in iteration:
        W = sgd(x, y, W0, df, 1e-2, i)
        success_t.append(train_validating(W))
        success_te.append(testing(W))
        print('finished one iter')
    alpha = np.arange(0, 1e-1, 1e-2)
    alpha_p = []
    for a in alpha:
        W = sgd(x, y, W0, df, a, 2000)
        alpha_p.append(testing(W))
        print("finished one iter")
    plt.step(iteration, success_t, label="training set")
    plt.step(iteration, success_te, label="testing set")
    plt.ylabel('accuracy')
    plt.xlabel('iterations')
    plt.legend(["training set", "testing set"], loc='best')
    plt.title("SGD learning curve alpha=1e-2")
    plt.savefig("SGD.png")
    plt.gca().clear()
    print("sgd learning curve")
    plt.plot(alpha, alpha_p)
    plt.ylabel('accuracy')
    plt.xlabel('alpha')
    plt.title("SGD learning rate vs accuracy iteration=2000")
    plt.savefig("SGDaccuracy.png")
    plt.gca().clear()
    print("alpha vs accu")
    """display weights"""
    W=sgd(x,y,W0,df,1e-2,1000)
    imsave('0.png', np.resize(W[:, 0].T[1:], (28, 28)))
    imsave('1.png', np.resize(W[:, 1].T[1:], (28, 28)))
    imsave('2.png', np.resize(W[:, 2].T[1:], (28, 28)))
    imsave('3.png', np.resize(W[:, 3].T[1:], (28, 28)))
    imsave('4.png', np.resize(W[:, 4].T[1:], (28, 28)))
    imsave('5.png', np.resize(W[:, 5].T[1:], (28, 28)))
    imsave('6.png', np.resize(W[:, 6].T[1:], (28, 28)))
    imsave('7.png', np.resize(W[:, 7].T[1:], (28, 28)))
    imsave('8.png', np.resize(W[:, 8].T[1:], (28, 28)))
    imsave('9.png', np.resize(W[:, 9].T[1:], (28, 28)))
    print("weights")
    """sgd_momentum"""
    success_tm = []
    success_tme = []
    for i in iteration:
        W = sgd_momentum(x, y, W0, df, 1e-2 * (1 - 0.99), i)
        success_tm.append(train_validating(W))
        success_tme.append(testing(W))
        print('finished one iter')
    plt.step(iteration, success_tm, label="training set")
    plt.step(iteration, success_tme, label="testing set")
    plt.ylabel('accuracy')
    plt.xlabel('iterations')
    plt.legend(["validating set", "training set", "testing set"], loc='best')
    plt.title("SGD momentum learning curve alpha=1e-2*(1-0.99)")
    plt.savefig("SGDm.png")
    plt.gca().clear()
    print("sgd momentum learning curve")

    """Contour"""
    W = sgd_momentum(x, y, W0, df, 1e-2 * (1 - 0.99), 700)
    gd_traj = get_traj(W, x, y, 10, momentum=False)
    mo_traj = get_traj(W, x, y, 10, momentum=True)
    gd_traj_1 = get_traj_1(W, x, y, 10, momentum=False)
    mo_traj_1 = get_traj_1(W, x, y, 10, momentum=True)
    w1s = np.arange(-10, 10, 1)
    w2s = np.arange(-10, 10, 1)
    w1z, w2z = np.meshgrid(w1s, w2s)
    C = np.zeros([w1s.size, w2s.size])
    for i in range(len(w1s)):
        for j in range(len(w2s)):
            C[j, i] = get_loss(w1s[i], w2s[j], W, x, y)
    plt.contour(w1z, w2z, C, cmap=cm.coolwarm)
    ln1, = plt.plot([a for a, b in gd_traj], [b for a, b in gd_traj], 'yo-', label="No Momentum 1")
    ln2, = plt.plot([a for a, b in mo_traj], [b for a, b in mo_traj], 'go-', label="Momentum 1")
    plt.legend(loc='best')
    plt.title('Contour plot')
    plt.xlabel('w1')
    plt.ylabel('w2')
    plt.savefig('Contour1.png')
    ln1.remove()
    ln2.remove()
    plt.plot([a for a, b in gd_traj_1], [b for a, b in gd_traj_1], 'yo-', label="No Momentum 2")
    plt.plot([a for a, b in mo_traj_1], [b for a, b in mo_traj_1], 'go-', label="Momentum 2")
    plt.savefig('Contour2.png')
    plt.gca().clear()
    print("contour plot 1&2")

    """part7"""
    start = timer()
    _ = sgd(x, y, W0, df, 1e-2, 10)
    backp_end = timer() - start
    _ = gradient_descend(x, y, W0, 1e-2, 10)
    end = timer() - start
    everyw_end = end - backp_end
    print('back propagation finished in:', backp_end)
    print('update-every-weight finished in:', everyw_end)
    print('accelerating ratio:', everyw_end / (backp_end * 1.0))
