import torch
import numpy as np

from matplotlib import pyplot as plt

from numpy import pi as PI



def gen_spiral(number, circles = 1, r0 = 0):
    '''
    Generates a 2-dimensional 2-class spiral dataset with variable complexity of the spirals. (No noise)

    Args:
        number (int): number of generated data per class
        circles (float or int): describes the complexity of the spirals, so the number of rotations of the data. default value 1. Also 0.5 etc. is accepted.
        r0 (float): specifies how narrow/close the spiral is around zero. When value is zero, the spiral is closed at the center. default 0.

    Out:
        two torch tensorsof shape (number,2) of the generated red and blue data respectively.
    '''
    res_red = torch.zeros((number,2))
    res_blue = torch.zeros((number,2))
    
    phi = torch.tensor(0.) # start value of phi
    delta_phi = torch.tensor(2 * PI * circles / number) # increment which is added to phi for each new datapoint
    
    r = r0 # initial radius
    delta_r = circles / number
    
    
    for i in range(number):
        x, y = (torch.cos(phi), torch.sin(phi))
        red = r * torch.tensor((x,y)) # generate red datapoint
        blue = r * torch.tensor((-x,-y)) # generate blue datapoint
        res_red[i].copy_(red)
        res_blue[i].copy_(blue)
        
        phi += delta_phi # updated phi
        r += delta_r  #updated radius

    return res_red, res_blue

def plot_spiral(red, blue, **kwargs):
    '''
    Generates scatterplot of the delivered data.

    Args:
        red (torch tensor of shape (n,2)): red datapoints
        blue (torch tensor of shape (m,2)): blue datapoints
        kwargs

    Out:
        no output. generates plot.
    '''
    plt.scatter(red[:,0], red[:,1], color =  'r', **kwargs)
    plt.scatter(blue[:,0], blue[:,1], color =  'b', **kwargs)