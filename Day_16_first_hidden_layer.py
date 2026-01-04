import numpy as np

# input data
input = np.array([.5, .3, .2])

# 4 neurons with 3 weights each
weight_layer = np.array([
    [.1, .2, .3],
    [.4, .5, .6],
    [.7, .8, .9],
    [-.1, -.2, -.3]
])

# baise
baises = np.array([.1, .2, -.1, .0])

# pre_activation
pre_activation  =   weight_layer @ input + baises

# Relu activation function 
def relu(x):
    return np.maximum(0, x)

output = relu(pre_activation)
print(output)
print(output.shape)