import numpy as np

# input data
input = np.array([.5, .3, .2])

# 4 neurons with 3 weights each
layer_weights = np.array([
    [.1, .2, .3],
    [.4, .5, .6],
    [.7, .8, .9],
    [-.1, -.2, -.3]
])

# baise
baises = np.array([.1, .2, -.1, .0])

# pre_activation
pre_activation = (layer_weights @ input) + baises

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


####### create the loss function
def binary_cross_entropy(y_true:int , y_pred:float):
    # add epsilon to prevent log(0)
    epsilon = 1e-15
    # clip the predicted value between epsilon and 1 - epsilon
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

y_true = 0

loss = binary_cross_entropy(y_true, output)

    

    
# gradient descent update rule 
# new_weights = weight - learning_rate  * derivative of the loss with respect weights (gradient)
learning_rate = 0.1
gradient = .8
new_output_weights = output_weights - (learning_rate * gradient)
print(f"output weight is:\n{output_weights}\nnew output weight:\n{new_output_weights}")
new_layer_weights = layer_weights - (learning_rate * gradient)
print(f"layer weight is:\n{layer_weights}\nnew layer weight:\n{new_layer_weights}")

