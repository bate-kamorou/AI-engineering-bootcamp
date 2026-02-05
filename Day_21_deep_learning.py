import os 
from keras import layers, Sequential
from Day_6_data_cleaner import DataCleaner
from sklearn.model_selection import train_test_split
# plotting 
import matplotlib.pyplot as plt
# import dropout
from keras.layers import Dropout
# import earlystopping and  model checkpointes
from keras.callbacks import EarlyStopping, ModelCheckpoint

TF_ENABLE_ONEDNN_OPTS=0

# import and clean the titanic data
data = DataCleaner("data/raw/titanic.csv")

df = data.clean_all("Age", "median",
                     "Fare", 
                    ["Sex", "Embarked"], 
                    ["Name", "Ticket", "Cabin","PassengerId","SibSp" ,"Parch"],
                    is_training=True)

# save the scaler
# data.save_scaler("data/processor/nn_scaler.joblib")

# # check if the processed data is already saved if not save it 
# if not os.path.exists("data/processed/cleaned_titanic.csv"):
#     # save the processed data to csv
#     df.to_csv("data/processed/cleaned_titanic.csv", index=False)

print(df.head(2))
# # Features and labels
X = df.drop(columns="Survived")
y = df["Survived"]


# split into train and test sets
X_train_val , X_test,  y_train_val, y_test = train_test_split(X, y, test_size=.15, random_state=42)

# split into train and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=.15, random_state=43)


# save the X_test and y_test dataset to be used for evaluation on the model later on

# check if the X_test split is already saved not save it 
# if not os.path.exists("data/processed/X_test.csv"):
#    X_test.to_csv("data/processed/X_test.csv", index=False)

# # check if the y_train split is already saved if not save it
# if not os.path.exists("data/processed/y_test.csv"):
#     y_test.to_csv("data/processed/y_test.csv", index=False)


#  instantiate the model
model = Sequential()

# get the input shape of the data
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
    patience=15,
    restore_best_weights=True,
)



# check if the best model is already saved so we don't have to overwrite the best model unless we want to
# by changing the filepath 

if not os.path.exists("models/best_titanic_removed_nn_model.keras"):
    # implente model checkpoints if model is not saved
    check_points = ModelCheckpoint(
        filepath="models/best_titanic_removed_nn_model.keras",
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
history = model.fit(X_train_val, 
                    y_train_val, 
                    batch_size=32, 
                    epochs=300, 
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop, check_points])

# evaluate the model loss and metrics on the test date
score = model.evaluate(X_test, y_test)
print(f"Best model loss is:\n{score[0]:.4} \nBest accuracy is :\n{score[1]:.4}")

# plot the train and validation loss with matplotlib

plt.plot(history.history["loss"], label="training loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.title("model loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

print(X_test.head(2))
print(y_test.head(2))
