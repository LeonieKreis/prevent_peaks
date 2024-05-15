from torch import nn

def feed_forward(hidden_layers=1, dim_hidden_layers=3, act_fun=nn.ReLU, type='fwd'):
    '''
    Generates feedforward model.

    Args:
        hidden_layers (int): default 1, specifies the number of hidden layers in the model.
        dim_hidden_layers (int or list): specifies the widths of the hidden layers.
        act_fun: default nn.ReLU, but any (for layer insertion suitable) activation function, implemented such that it can be used in sequential model construction (must be wrapped as a module).

    Out: 
        sequential pytorch model. If input is invalid, raise ValueError.
    '''
    modules = []
    if hidden_layers == 0:  # if no hidden layers are specified, the net is a linear model
        return nn.Sequential(nn.Linear(2, 2))
    if isinstance(dim_hidden_layers, int):  # all hidden layers have the same width
        dim_hidden_layers = [dim_hidden_layers] * hidden_layers
    # the hidden layers have different widths
    if isinstance(dim_hidden_layers, (list, tuple)):
        dim_old = 2

        for dim in dim_hidden_layers:
            dim_curr = dim
            modules.append(
                nn.Linear(in_features=dim_old, out_features=dim_curr))
            modules.append(act_fun())
            dim_old = dim_curr

        modules.append(nn.Linear(dim_old, 2))

        return nn.Sequential(*modules)

    raise ValueError()