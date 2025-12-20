import pandas as pd

# read CSV file into DataFrame
df = pd.read_csv("titanic.csv")

# display first 5 rows of the DataFrame
# print(df.head())

# display summary statistics of the DataFrame
# print(df.describe())

# get information about the DataFrame
# print(df.info())

# print the colunmns names
print("column names:", df.columns.tolist())

# len of passengers in the dataset that are less than 18 years old
num_passengers_under_18 = len(df[df["Age"] < 18])
print("Number of passengers under 18 years old:", num_passengers_under_18)

# create a dateframe containing only the survived passengers
survivors_df = df[df["Survived"] == 1]
print("len of survivors_df:",len(survivors_df))

# create a dateframe containing only the non-survived passengers
non_survivors_df = df[df["Survived"] == 0]
print("len of non_survivors_df:",len(non_survivors_df))

# find the average age of survivors
survivors_avg_age = survivors_df["Age"].mean()
print("Average age of survivors:", survivors_avg_age)

# find the average age of non-survivors
non_survivors_df_avg_age = non_survivors_df["Age"].mean()
print("Average age of non-survivors:", non_survivors_df_avg_age)

# find the number of survivors that are less than 18 years old
num_survivors_under_18  = len(survivors_df[survivors_df["Age"] < 18])
print("Number of survivors under 18 years old:", num_survivors_under_18)

# find out how many passenngers were in each class (Pclass)
passenger_per_class = df["Pclass"].value_counts()
print("Number of passengers per classs:\n",passenger_per_class)

# create a new column "isChild" that is True if Age < 18 else false
df["isChild"] = df["Age"] < 18
print(df[["Age", "isChild"]].head())