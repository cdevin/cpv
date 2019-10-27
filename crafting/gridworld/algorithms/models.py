from torch import nn, optim

def layer_init(layer):
    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    nn.init.constant_(layer.bias.data, 0)
    return layer
