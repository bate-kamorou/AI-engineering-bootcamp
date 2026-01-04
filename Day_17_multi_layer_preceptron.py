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
pre_activation = (weight_layer @ input) + baises

# Relu activation function 
def relu(x):
    return np.maximum(0, x)

# first layer output
hidden_layer_output = relu(pre_activation)

# output weight
output_weights = np.array([.5,-.2,.3,.1])

# output bais
output_bias = .2

# weighted output
weighted_output  = hidden_layer_output @ output_weights + output_bias

# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# final output
output = sigmoid(weighted_output)

print(output)




