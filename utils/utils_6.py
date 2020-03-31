import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def load_data(filepath):
    with np.load(filepath) as f:
        patterns = f['patterns']
        return (patterns)
    
class Animation:
    def __init__(self,num_rows):
        self.num_rows = num_rows
        self.fig, self.ax = plt.subplots()
        plt.close()

    def init_func(self):
        self.ax.set_xlim([-.5,self.num_rows-0.5])
        self.ax.set_ylim([-.5,self.num_rows-0.5])
        a = np.zeros((self.num_rows,self.num_rows))
        a[1::2,::2] = 1
        a[::2,1::2] = 1
        self.ax.imshow(a,cmap='gray')   
        self.queens_plot, = self.ax.plot([],[],'*y',markersize=14)
        return (self.queens_plot,)

    def func(self, t):
        queen = t
        self.queens_plot.set_data(queen[0],queen[1])
        return (self.queens_plot,)

    def play(self, frames):
        anim = animation.FuncAnimation(self.fig, 
                                       func=self.func,
                                       frames=frames,
                                       init_func=self.init_func,
                                       blit=True)
        return HTML(anim.to_jshtml(default_mode='once'))

def hopfield(patterns, hopfieldInitWeights, hopfieldAssociate, noise_level, offState):
    # Inputs: patterns      Cell array of patterns (images)
    #         hopfieldInitWeights function to initiate weights
    #         hopfieldAssociate   function to determine final activation
    #         noise_level   Amount of noise to add to the patterns for testing, given as a
    #                       fraction of the number of pixels, i.e. noise_level=0.1 flips
    #                       ten percent of the pixels.
    num_patterns  = len(patterns)
    num_inputs = patterns[0].shape
    
    ims = []
    fig = plt.figure(figsize=(9, 25))
    # Initialize weights
    weights = hopfieldInitWeights(patterns)

    for num in range(num_patterns):
        print('Trying to associate pattern {:d}\n'.format(num))
    
        # Plot original pattern
        plt.subplot(num_patterns,3,3*num+1)
        plt.imshow(patterns[num])
        plt.title('original pattern')
    
        # Add noise to the pattern
        noise = (np.random.rand(num_inputs[0],num_inputs[1]) < noise_level)
        activation = patterns[num].copy()
        activation[noise] = - activation[noise]
    
        # Plot noisy pattern
        plt.subplot(num_patterns,3,3*num+2)
        plt.imshow(activation)
        plt.title('pattern with noise')
    
        # Try to recognize pattern
        plt.subplot(num_patterns,3,3*num+3)
        activation = hopfieldAssociate(weights, activation,offState)
        plt.imshow(activation)
        plt.title('hopfield fixpoint')
    
def hopfieldAssociateQueens(weights, activation,offState):

    # Threshold for neurons
    threshold = 0;
    
    original_shape = activation.shape
    
    # make activation a vector
    activation = activation.flatten()

    N = len(activation)
    
    # check if changes occur    
    change = False

    for index in np.random.permutation(N):
        # update
        old = activation[index]
        activ = weights[index]@activation
        activation[index] = 1*(activ >= threshold) + offState *(activ<threshold)
        if not (old == activation[index]):
            change = True
            
    activation = np.reshape(activation,original_shape)
    
    return activation, change

def hopfield_queen(initQueens, num_rows):
# Inputs: InitQueens  function to initiate weights

    weights = initQueens(num_rows)
    activation = np.zeros((num_rows,num_rows))
    
    change = True
        
    while change:
        
        activation, change = hopfieldAssociateQueens(weights, activation,0)
        
        if (not change) and np.sum(activation) < num_rows:
            change = True
            activation[np.random.randint(num_rows)] = 0
        
        yield np.where(activation)
        
    

