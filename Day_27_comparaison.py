from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from Day_6_data_cleaner import DataCleaner


# # import the processd data 
processed_data = pd.read_csv("data/processed/titanic_feature_engineered_removing.csv")
print(processed_data.head())


# use the unproceesed data it can be processede and feature engineered and
# see if it's improve the model

# get the raw data
data_path = "data/raw/titanic.csv"

# instantiate the data cleaner class
# data = DataCleaner(data_path)

# processed_data = data.clean_all("Age", "median",
#                      "Fare", 
#                     ["Sex", "Embarked"], 
#                     ["Name", "Ticket", "Cabin","PassengerId"],
#                     is_training=True)


X = processed_data.drop(columns="Survived")
y = processed_data["Survived"]
# split the processed data into train and test 
X_train, X_val,y_train, y_val = train_test_split(X, y, test_size=.2)

# instantiate the random  regressort
rand_forest_r = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    n_jobs=-1
)

# train the random forest
rand_forest_r.fit(X_train, y_train)

# save the model
joblib.dump(rand_forest_r, "models/rand_forest_cls_1.joblib")

# random forest metrics on the test set
acc = rand_forest_r.score(X_val, y_val)
print(f"Random forest acccuracy: {acc:2f}")


# make a prediction
predictions = rand_forest_r.predict(X_val)

# confusion matrix
conf_mat = confusion_matrix(y_val, predictions )
print(f"confusion matrix: \n {conf_mat}")


# random forest classification report
class_report = classification_report(y_val, predictions)
print(f"classification report :\n {class_report}")
# 