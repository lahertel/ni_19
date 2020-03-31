import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

def load_data(filepath):
    with np.load(filepath) as f:
        samples = f['samples']
        return samples

def plot_data(samples):
    fig, ax = plt.subplots()
    ax.plot(samples[:,0], samples[:, 1], 'b.')
    ax.set_aspect('equal')

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0,
                    shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

def plot_principal_components(samples, eigenvectors=None):
    plot_data(samples)
    if eigenvectors is not None:
        mean = np.mean(samples, axis=0)
        for eigenvector in eigenvectors.T:
            draw_vector(mean, mean + eigenvector)

def plot_joint(samples, title=None):
    fig = plt.figure()

    if title:
        fig.suptitle(title, size=18)

    gs = GridSpec(4, 4)

    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0, 0:3])
    ax_marg_y = fig.add_subplot(gs[1:4, 3])

    x = samples[:, 0]
    y = samples[:, 1]
    
    ax_joint.plot(x, y, 'b.')
    ax_marg_x.hist(x,
                   bins=25)
    ax_marg_y.hist(y,
                   bins=25,
                   orientation="horizontal")

    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_x.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

def rotation_matrix(theta):
    theta = np.radians(theta)
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array(((cos, -sin),
                     (sin, cos))).T

def kurtosis(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    z = (x - mean) / std
    return np.mean(z ** 4, axis=0)
