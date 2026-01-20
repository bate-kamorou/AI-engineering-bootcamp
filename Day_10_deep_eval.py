from  Day_6_data_cleaner import DataCleaner
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot  as plt
# import for the deep evaluation
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
df = DataCleaner("data/raw/titanic.csv")

cols = ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
       'Sex_male', 'Embarked_Q', 'Embarked_S']

# preprocess the data
df.handle_missing_data("Age", "median")

df.min_max_scale("Fare")

df.encode_categorical(["Sex","Embarked"])

df.remove_columns(["Name", "Ticket", "Cabin"])

df = df.get_clean_data()

# print(df.head())

# Spilt the data into X and y
X = df.drop(columns=["Survived"])
y = df["Survived"]

# split into train adn test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# instantiate the classifier
cls = LogisticRegression(max_iter=1000)

# fit the classifier
cls.fit(X_train, y_train)

# make perdictions
y_preds = cls.predict(X_test)

# score the model on the test
accuracy = cls.score(X_test, y_test)
print("classifier accuracy is :", accuracy)


######## ------------ 
# deep evaluation

# confusion matrix
conf_mat = confusion_matrix(y_test, y_preds, labels=[0, 1])
print(conf_mat)
# output of the conf_mat = [[89 16][19 55]]

# classification report
cls_report = classification_report(y_test, y_preds)
print(cls_report)
# classification report output 
#   precision    recall  f1-score   support

#            0       0.82      0.85      0.84       105
#            1       0.77      0.74      0.76        74

#     accuracy                           0.80       179
#    macro avg       0.80      0.80      0.80       179
# weighted avg       0.80      0.80      0.80       179

# ConfusionMatrixDisplay
conf_mat_display = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=cls.classes_)
conf_mat_display.plot()
plt.show()