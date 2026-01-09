import os 
import tensorflow as tf
from keras import layers, Sequential, models
from Day_6_data_cleaner import DataCleaner
from sklearn.model_selection import train_test_split
#plotting 
import matplotlib.pyplot as plt
# import dropout
from keras.layers import Dropout
# import earlystopping and  model checkpointes
from keras.callbacks import EarlyStopping, ModelCheckpoint

TF_ENABLE_ONEDNN_OPTS=0
# import and clean the titanic data
data = DataCleaner("data\\raw\\titanic.csv")
df = data.clean_all("Age", "median", "Fare", ["Sex", "Embarked"], ["Name", "Ticket", "Cabin","PassengerId"])

print(df.head(2))
# # Features and labels
X = df.drop(columns="Survived")
y = df["Survived"]


# split into train and test sets
X_train_val , X_test,  y_train_val, y_test = train_test_split(X, y, test_size=.15, random_state=42)

# split into train and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=.15, random_state=43)
# # instantiate the model

model = Sequential()

input_shape =  X.shape[1]
# # add a the input layer
model.add(layers.Input((input_shape,)))

# add a first hidden layer with 16 neurons and relu activation function
model.add(layers.Dense(16, activation="relu"))

# add 20% dropout to active neurons
model.add(Dropout(0.2))
# add a second hidden layer  with 12 neurons and  relu activation function
model.add(layers.Dense(12, activation="relu"))

# add a third hidden layer with 6 neurons and relu activation function
model.add(layers.Dense(6, activation="relu"))

# add an output layer with 1 neuron and a sigmoid function 
model.add(layers.Dense(1, activation="sigmoid"))

# implente the earlystopping 
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)


# check in the models dir exsist if not create one
if not os.path.exists("models"):
    os.makedirs("models")


# implente model checkpoints
check_points = ModelCheckpoint(
    filepath="models\\best_titanic_nn1.keras",
    monitor="val_loss",
    save_best_only=True,
)
# complie the model with optimizer , loss and metrics
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# train the model 
history = model.fit(X_train, 
                    y_train, 
                    batch_size=32, 
                    epochs=500, 
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop, check_points])

# evaluate the model loss and metrics on the test date
score = model.evaluate(X_test, y_test)
print(f" Best model loss is:\n{score[0]:.4} \nBest accuracy is :\n{score[1]:.4}")
# Best model loss is:
# 0.4618
# Best accuracy is :
# 0.806
# at 151 of 500 training epochs




# plot the train and validation loss with matplotlib

plt.plot(history.history["loss"], label="training loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.title("model loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


