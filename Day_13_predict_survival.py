import joblib
import pandas as pd

# load the modelb
rand_cls = joblib.load("models\\titanic_final_model.joblib")
print("model successfully loaded")

# create a data to make a prediction with
fake_passenger_data = [[1, 45.5, 0, 0, 0.055628, 1, 0, 1]]
feature_names = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
fake_passenger = pd.DataFrame(fake_passenger_data, columns=feature_names)

prediction = rand_cls.predict(fake_passenger)
print("model's prediction on the fake passenge",prediction)