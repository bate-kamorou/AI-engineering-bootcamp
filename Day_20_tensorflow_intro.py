# imports
import tensorflow as tf
import keras
from keras import layers, models, Sequential

# initialize the  model
model = Sequential()
# input layer 
model.add(keras.Input((3,)))
# Add the hidden layer
model.add(layers.Dense(4, activation="relu"))
# Add an output layer"
model.add(layers.Dense(1, activation="sigmoid"))
# Loss and optimizer
model.compile(
    optimizer="sgd",
    loss="binary_crossentropy",
    metrics=["accuracy"]

)

print(model.summary())