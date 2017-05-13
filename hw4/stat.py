import numpy as np
import scipy.linalg as li
from sklearn.metrics.pairwise import euclidean_distances
#import matplotlib.pyplot as plt

def elu(arr):
    return np.where(arr > 0, arr, np.exp(arr) - 1)


def make_layer(in_size, out_size):
    w = np.random.normal(scale=0.5, size=(in_size, out_size))
    b = np.random.normal(scale=0.5, size=out_size)
    return (w, b)


def forward(inpd, layers):
    out = inpd
    for layer in layers:
        w, b = layer
        out = elu(out @ w + b)

    return out


def gen_data(dim, layer_dims, N):
    layers = []
    data = np.random.normal(size=(N, dim))

    nd = dim
    for d in layer_dims:
        layers.append(make_layer(nd, d))
        nd = d

    w, b = make_layer(nd, nd)
    gen_data = forward(data, layers)
    gen_data = gen_data @ w + b
    return gen_data


if __name__ == '__main__':
    stdlist = []
    maxdim = 60
    for i in range(1,maxdim+1):
        dim = i   # intrinsic dimension of i
        N = 10000  # num data points
        # the hidden dimension is randomly chosen from [60, 79] uniformly
        layer_dims = [np.random.randint(60, 80), 100]
        data = gen_data(dim, layer_dims, N)
        # (data, dim) is a (question, answer) pair
        dist = euclidean_distances(data)
        dist.partition(1, axis=1)
        dist_min = dist[:,1]
        std = np.std(dist_min)
        print('{}: {}'.format(i,std))
        stdlist.append(std)
    x = np.log([i for i in range(1,maxdim+1)])
    stdlist = np.array(stdlist)
    w = np.polyfit(x,stdlist,deg=1)
    print(w)

    with open('stat.csv','w')as csvfile:
        csvfile.write(str(w[0])+','+str(w[1]))

    '''fig = plt.figure()
    fig.suptitle('std-to-dim(N={})'.format(N))
    ax1 = fig.add_subplot(111)
    ax1.plot(x,stdlist)
    fig.savefig('{}.png'.format(N))'''
