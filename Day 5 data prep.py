import pandas as pd

# read the csv file into a Dataframe
df = pd.read_csv("titanic.csv")

# check for missing values in each column
missing_values_per_col = df.isnull().sum()
# print("Missing values per column:\n", missing_values_per_col)

# get the mean of the 'Age' column, ignoring missing values
mean_age = df["Age"].mean()

# fill the missing values of the "Age" column with the mean age
df["Age"] = df["Age"].fillna(mean_age)


# fill in the missing values of the "Embarked" column with the mode (most common value)
embarked_mode = df["Embarked"].mode()[0]
# print(embarked_mode)

# fill the missing values of the "Embarked" column with the model
df["Embarked"].fillna(embarked_mode, inplace=True)
# check again for missing values to confirm
missing_values_per_col_after_filling = df.isnull().sum()
# print("Missing values per column after filling 'Age':\n", missing_values_per_col_after_filling)


# one hot encoding
one_hot_encode_cols = ["Sex", "Embarked"]
df = pd.get_dummies(df, columns=one_hot_encode_cols, drop_first=True, dtype=int)
# print(df.columns.tolist())
# print(df["Sex_male"].head())
# print(df["Embarked_Q"].head())
# normalization and standardization 

# manual scaling on the "Fare" cloumn 
import numpy as np
# get the fare column values and convert to numpy array
fare_values = df["Fare"].values
fare_values_array = np.array(fare_values)
print("fare values:", fare_values[:5])

# get max and min values of the fare column
fare_max = np.max(fare_values_array)
fare_min = np.min(fare_values_array)
print(fare_max, fare_min)

# scale the fare cloumn using min-max scaling
df["Fare"] = (fare_values_array - fare_min) / (fare_max - fare_min)
print("Scaled fare values:", df["Fare"].head())

# print(df.head)
print(df.dtypes[df.dtypes == object])



# remove unwanted columns
drop_cols = ["Name", "Ticket", "Cabin"]

df = df.drop(columns=drop_cols)
print(df.head())
print(df.dtypes)


