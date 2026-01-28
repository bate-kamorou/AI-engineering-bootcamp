import pandas as pd
from Day_6_data_cleaner import DataCleaner

d = DataCleaner("data/raw/fake_data.csv")

p =  d.clean_all("Age", "median",
                     "Fare", 
                    ["Sex", "Embarked"], 
                    ["Name", "Ticket", "Cabin","PassengerId"],
                    is_training=True)


print(p)
      
