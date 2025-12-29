from Day_6_data_cleaner import DataCleaner
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# load the data
df = DataCleaner("titanic.csv")

# handle missing values
df.handle_missing_data("Age", "median")
# scale the fare column
df.min_max_scale("Fare")
# encode  one hot encode the sex ans embarked column
encode_cols = ["Sex", "Embarked"]
df.encode_categorical(encode_cols)
#remove the unnecessary column
remove_cols = ["Name", "Ticket", "Cabin"]
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
rand_f_cls = RandomForestClassifier(random_state=42)

# fit the on the training set
rand_f_cls.fit(X_train, y_train)

# test the model on the test set
score = rand_f_cls.score(X_test, y_test)

print("model accuracy is :", score)
# model accuracy is : 0.8268156424581006

# make a prediction
y_preds = rand_f_cls.predict(X_test)

# deep evaluation of the random forest model
# confusion matrix
conf_mat = confusion_matrix(y_test, y_preds, labels=rand_f_cls.classes_)
# classification report
classfi_report = classification_report(y_test, y_preds)
# print(classfi_report)

conf_display = ConfusionMatrixDisplay(conf_mat, display_labels=rand_f_cls.classes_)
conf_display.plot()
plt.title("confusion matrix display")
plt.show()

# feature importance
important_features = pd.Series(rand_f_cls.feature_importances_, index=X.columns)
important_features.sort_values().plot(kind="barh")
plt.title("Most important features")
plt.show()

# random forest parameters optimization]
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import 
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10]
}

# instantiate the grid search cross validation
grid_search_cv  = GridSearchCV(estimator=rand_f_cls, 
                               param_grid=param_grid,
                               scoring="accuracy",
                               n_jobs=-1, 
                               cv = 5,
                            )

grid_search_cv.fit(X_train, y_train)

params = grid_search_cv.best_params_
print(params)


# random forest with the best parameters
best_cls = grid_search_cv.best_estimator_

# make prediction with the classifier 
best_preds = best_cls.predict(X_test)

best_cls_score = best_cls.score(X_test, y_test)
print("accuracy : ",best_cls_score)

# the random classifier tuned on the best parameters from  the gride search cv
# output is {'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 100}
# accuracy :  0.8212290502793296

# classification report of the classifier with best parameters
best_cls_report = classification_report(y_test, best_preds)
print("classification report of the best rand classifier:", best_cls_report)

# # output 
# classification report of the best rand classifier:               precision    recall  f1-score   support

#            0       0.81      0.90      0.86       105
#            1       0.84      0.70      0.76        74

#     accuracy                           0.82       179
#    macro avg       0.83      0.80      0.81       179
# weighted avg       0.82      0.82      0.82       179
