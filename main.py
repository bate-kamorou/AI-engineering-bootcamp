# # import pandas as pd
# # import sys 

# # sys.path.append =  ["Day-6-capstone-project-(Data_cleaner)"]

from Day_6_capstone_project_data_cleaner import DataCleaner

# # test the class

# create an instance of the DataCleaner class with the titanic dataset
titanic_data = DataCleaner("titanic.csv")

titanic_data.handle_missing_data("Age", "mean")
titanic_data.min_max_scale("Fare")

encode_cols = ["Sex", "Embarked"]
titanic_data.encode_categorical(encode_cols)

remove_cols = ["Name", "Ticket", "Cabin"]
titanic_data.remove_columns(remove_cols)

print(titanic_data.get_clean_data())
print(titanic_data.df.isna().sum())