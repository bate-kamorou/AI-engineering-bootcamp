import pandas as pd

# read the csv file into a Dataframe
df = pd.read_csv("data/raw/titanic.csv")

# check for missing values in each column
missing_values_per_col = df.isnull().sum()
# print("Missing values per column:\n", missing_values_per_col)

# get the mean of the 'Age' column, ignoring missing values
median_age = df["Age"].median()

# fill the missing values of the "Age" column with the mean age
df["Age"] = df["Age"].fillna(median_age)


# fill in the missing values of the "Embarked" column with the mode (most common value)
embarked_mode = df["Embarked"].mode()[0]
# print(embarked_mode)

# fill the missing values of the "Embarked" column with the model
df["Embarked"] = df["Embarked"].fillna(embarked_mode)
# check again for missing values to confirm
missing_values_per_col_after_filling = df.isnull().sum()
# print("Missing values per column after filling 'Age':\n", missing_values_per_col_after_filling)


# one hot encoding
one_hot_encode_cols = ["Sex", "Embarked"]
df = pd.get_dummies(df, columns=one_hot_encode_cols, drop_first=True, dtype=int)
# print(df.columns.tolist())

# manual scaling on the "Fare" cloumn 

# get the fare column values and convert to numpy array
fare_values = df["Fare"].values

# scale the fare cloumn using min-max scaling
df["Fare"] = (df["Fare"] - df["Fare"].min()) / (df["Fare"].max() - df["Fare"].min())
print("Scaled fare values:", df["Fare"].head())

# remove unwanted columns
drop_cols = ["Name", "Ticket", "Cabin"]

df = df.drop(columns=drop_cols)
print(df.head())
print(df.dtypes[df.dtypes == object])
