import numpy as  np

# weight
weight =  np.array([0.4, 0.7, 0.8])
# input
input = np.array([0.5, 0.3, .2])
# bais
bias = -0.5

# activation function
def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

# preactivation weight  * input + bias
pre_activation =  np.dot(input, weight) + bias

# neuron output 
output = sigmoid(pre_activation)
print(output)
