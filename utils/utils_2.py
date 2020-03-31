import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML

class Animation:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.fig, self.ax = plt.subplots()
        plt.close()

    def init_func(self):
        features_pos = self.features[self.labels == 1]
        features_neg = self.features[self.labels == -1]
        self.ax.plot(features_pos[:,0], features_pos[:, 1], 'bo')
        self.ax.plot(features_neg[:,0], features_neg[:, 1], 'rx')
        self.ax.axis('scaled')
        self.classline, = self.ax.plot([], [], lw=2)
        return (self.classline,)

    def func(self, t):
        weights, threshold = t
        x_min, x_max = self.ax.get_xbound()
        y_min, y_max = self.ax.get_ybound()
        if weights[1] == 0:
            x_min = threshold / weights[0]
            x_max = x_min
        else:
            y_min = (threshold - weights[0] * x_min) / weights[1]
            y_max = (threshold - weights[0] * x_max) / weights[1]
        self.classline.set_data([x_min, x_max], [y_min, y_max])
        return (self.classline,)

    def play(self, frames):
        anim = animation.FuncAnimation(self.fig, 
                                       func=self.func,
                                       frames=frames,
                                       init_func=self.init_func,
                                       blit=True)
        return HTML(anim.to_jshtml(default_mode='once'))


def load_data(filepath):
    with np.load(filepath) as f:
        samples = f['samples']
        labels = f['labels']
        return (samples, labels)
