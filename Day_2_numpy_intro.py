import numpy as  np 

# create input array
input = np.array([1.0, 2.0, 3.0,])

# create weight and bias arrays
weight  = np.random.randn(3,2)

bias = np.array([0.5, 0.1])

# perform layer calculation  
output = np.dot(input, weight) + bias

  
# create relu activation function
def relu(x):
    if x < 0:
        return 0
    else:
        return x


# test relu function on output array
relu_output = np.array([relu(item) for item in output])


# perform relu using numpy maximum function
output_2  = np.maximum(0, output)
print("Output using relu function:", relu_output)   
print("Output using numpy maximum function:", output_2)
print("Are both outputs equal?", np.array_equal(relu_output, output_2))

# perform layer calculation using @ operator
out = input @ weight + bias
print("Output using @ operator:", out)
print("Are outputs using np.dot and @ operator equal?", np.array_equal(output, out))