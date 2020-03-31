import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def load_data(filepath):
    with np.load(filepath) as f:
        samples = f['samples']
        labels = f['labels']
        return (samples, labels)

def sigmoid(x, beta):
    return np.tanh(0.5 * beta * x)

def extend(tensor):
    if tensor.ndim == 1:
        return np.insert(tensor, 0, -1)
    else:
        thresholds = -np.ones(len(tensor))
        return np.column_stack((thresholds, tensor))

def classify_mlp(samples, hidden_weights, output_weights, beta):
    samples = extend(samples)
    hidden_outputs = sigmoid(np.matmul(samples, hidden_weights), beta)
    hidden_outputs = extend(hidden_outputs)
    outputs = sigmoid(np.matmul(hidden_outputs, output_weights), beta)    
    return (outputs, hidden_outputs)

def plot_data(samples, labels):
    samples_pos = samples[labels == 1]
    samples_neg = samples[labels == -1]    
    plt.plot(samples_pos[:,0], samples_pos[:, 1], 'bo')
    plt.plot(samples_neg[:,0], samples_neg[:, 1], 'rx')
    plt.axis('scaled')

class Animation:
    def __init__(self, samples, labels, hidden_neurons):
        self.samples = samples
        self.labels = labels
        self.hidden_neurons = hidden_neurons
        self.fig, self.ax = plt.subplots()
        plt.close()

    def get_classlines(self, weights):
        for weight in weights.T:
            yield self.get_classline(weight[1:], weight[0])

    def get_classline(self, weights, threshold):
        assert weights.any(), "Weights must not be the zero vector."
        x_min, x_max = self.ax.get_xbound()
        y_min, y_max = self.ax.get_ybound()
        if weights[1] == 0:
            x_min = threshold / weights[0]
            x_max = x_min
        else:
            y_min = (threshold - weights[0] * x_min) / weights[1]
            y_max = (threshold - weights[0] * x_max) / weights[1]
        return ([x_min, x_max], [y_min, y_max])

    def init_func(self):
        self.pos = self.ax.plot([], [], 'bo')
        self.neg = self.ax.plot([], [], 'rx')
        self.classlines = [self.ax.plot([], [], lw=3)[0] for _ in range(self.hidden_neurons)]
        self.ax.set_xlim([-1.1, 1.1])
        self.ax.set_ylim([-1.1, 1.1])
        self.ax.axes.set_aspect('equal')
        return (self.pos + self.neg + self.classlines)

    def func(self, t):
        hidden_weights, output_weights = t
        classifications, _ = classify_mlp(self.samples, hidden_weights, output_weights, beta=2)
        classifications = 2 * (classifications >= 0) - 1;
        samples_pos = self.samples[classifications == 1]
        samples_neg = self.samples[classifications == -1]
        self.pos[0].set_data(samples_pos[:, 0], samples_pos[:, 1])
        self.neg[0].set_data(samples_neg[:, 0], samples_neg[:, 1])
        classlines = list(self.get_classlines(hidden_weights))
        for idx, classline in enumerate(classlines):
            self.classlines[idx].set_data(classline)
        return (self.pos + self.neg + self.classlines)

    def play(self, frames, step_size=1):
        frames = frames[0::step_size]
        anim = animation.FuncAnimation(self.fig, 
                                       func=self.func,
                                       frames=frames,
                                       init_func=self.init_func,
                                       blit=True)
        return HTML(anim.to_jshtml(default_mode='once'))
