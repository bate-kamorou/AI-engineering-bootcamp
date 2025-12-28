import pandas as pd
from Day_6_data_cleaner import DataCleaner
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# load the data
df = DataCleaner("titanic.csv")

# handle missing values
df.handle_missing_data("Age", "mean")
# scale the fare column
df.min_max_scale("Fare")
# encode  one hot encode the sex ans embarked column
encode_cols = ["Sex", "Embarked"]
df.encode_categorical(encode_cols)
#remove the unnecessary column
remove_cols = ["Name", "Ticket", "Cabin"]
df.remove_columns(remove_cols)

df = df.get_clean_data()

# print(df.columns)
cols = ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
       'Sex_male', 'Embarked_Q', 'Embarked_S']
# # features X and label y 
X = df.drop(columns=["Survived"])
y = df["Survived"]

# print(X.head())
# print(y.head())

# split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# instantiate the classifier
cls = linear_model.LogisticRegression(max_iter=1500)

# fit the on the training set
cls.fit(X_train, y_train)

# test the model on the test set
score = cls.score(X_test, y_test)

print("model accuracy is :", score)





