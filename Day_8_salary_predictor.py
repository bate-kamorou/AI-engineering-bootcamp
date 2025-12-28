from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

# Feature X
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]).reshape(-1 , 1)

# Label y 
y =  np.array([45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000])


# instantiate the model
model = linear_model.LinearRegression()

# train the model
model.fit(X,y)

# make a prediction 
pred = model.predict(np.array([[12]]))
print(pred)


# get the models weight and bias
weight = model.coef_
bias = model.intercept_

print(f"models weights are : {weight} \n model bias are : {bias}")

# plot the data
fig, ax = plt.subplots(1,1)
plt.scatter(X, y , color="blue", label="Actual label")
plt.plot(X, model.predict(X), color="red", label="model prediction")
plt.legend()
plt.show()
