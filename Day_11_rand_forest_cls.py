from Day_6_data_cleaner import DataCleaner
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# load the data
df = DataCleaner("data/raw/titanic.csv")

# handle missing values
df.handle_missing_data("Age", "median")
# scale the fare column
df.min_max_scale("Fare")
# encode  one hot encode the sex ans embarked column
encode_cols = ["Sex", "Embarked"]
df.encode_categorical(encode_cols)
#remove the unnecessary column
remove_cols = ["Name", "Ticket", "Cabin", ]
df.remove_columns(remove_cols)

df = df.get_clean_data()

# # features X and label y 
X = df.drop(columns=["Survived"])
y = df["Survived"]

# print(X.head())
# print(y.head())

# split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# instantiate the classifier
rand_f_cls = RandomForestClassifier(n_estimators=150, max_depth=5)

# fit the on the training set
rand_f_cls.fit(X_train, y_train)

# test the model on the test set
score = rand_f_cls.score(X_test, y_test)

print("model accuracy is :", score)

# make a prediction
y_preds = rand_f_cls.predict(X_test)

# deep evaluation of the random forest model
# confusion matrix
conf_mat = confusion_matrix(y_test, y_preds, labels=rand_f_cls.classes_)
# classification report
classfi_report = classification_report(y_test, y_preds)
print(classfi_report)

# feature importance
important_features = pd.Series(rand_f_cls.feature_importances_, index=X.columns)
important_features.sort_values().plot(kind="barh")
plt.title("Most important features")
plt.show()


print(X_train.head(1))



