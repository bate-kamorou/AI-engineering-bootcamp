from sklearn.model_selection import GridSearchCV, train_test_split 
from sklearn.metrics import classification_report
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from Day_6_data_cleaner import DataCleaner
import numpy as np
import matplotlib.pyplot as plt



# instantiate the data cleaner class
data_path = "data/raw/titanic.csv"

data = DataCleaner(data_path)

processed_data = data.clean_all("Age", "median",
                     "Fare", 
                    ["Sex", "Embarked"], 
                    ["Name", "Ticket", "Cabin","PassengerId", "SibSp","Parch" ],
                    is_training=True)

# save the processed data
processed_data.to_csv("data/processed/titanic_feature_engineered_removing.csv")

X = processed_data.drop(columns="Survived")
y = processed_data["Survived"]
# split the processed data into train and test 


X_train, X_val,y_train, y_val = train_test_split(X, y, test_size=.2)

# set the parameters to use for the grid search
params = {
    "n_estimators":[100, 150],
    "max_depth": [2,4],
    "min_samples_split": [2,4],
    'criterion': ['gini', 'entropy']
}

# load the model
rd_f = RandomForestClassifier()

# instantiate the search
gs = GridSearchCV(rd_f, param_grid=params, cv=4, n_jobs=-1, scoring="accuracy")
# fit the search
gs.fit(X, y)

print("Best estimator is \n", gs.best_estimator_)

print("Best parameters are :\n", gs.best_params_)

print("Best score achived is : \n" , gs.best_score_)

# fit the best estimator
best_estimator = gs.best_estimator_

best_estimator.fit(X_train, y_train)

# accuracy score of the best model
score = best_estimator.score(X_val, y_val)
print("Best random forest score on the validation set:", score)

# make a prediction 
y_preds = best_estimator.predict(X_val)

# save the model
joblib.dump(best_estimator, "models/best_rf_estimator.joblib")

# classification report 

cls_report = classification_report(y_val, y_preds)
print("classification_report: \n", cls_report)


# feature importance
important_features  = best_estimator.feature_importances_
features = X.columns
indices = np.argsort(important_features)

# plotting the feature importance
plt.title("Feature importance")
plt.barh(y=range(len(indices)), width=important_features[indices], align="center")
plt.yticks(ticks=range(len(indices)), labels=[ features[i] for i in indices])
plt.xlabel("Relative importance")
plt.show()