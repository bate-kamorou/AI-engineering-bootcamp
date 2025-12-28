import pandas as pd 

class DataCleaner:
    """ 
    Takes in a messy dataset, clean it and return a cleaned dataset

    Methods:

    handel_missing_data(column, strategy=mean)
        fill in missing values of a  column based on the given strategy
    
    min_max_scale(column)
         Applies min max scaling to the specifed column 

    encode_categorical(column)
        One hot encode a list of specified columns 

    remove_columns(column)
        remove unneccessary  columns forn the data set
    
    get_clean_data()
        Return the clean dataset

    """

    def __init__(self, file_path:str) : 
        # read the file into a dataframe
        """
        Takes in the path to the data that will be cleaned up
        
        args:

        file_path: the path to the data to be cleaned
        """
        self.df = pd.read_csv(file_path)
        print(f"Data loaded successfully. shape {self.df.shape}")


    def handle_missing_data(self, column:str,strategy:str = "mean") -> None:
        """
        fill in missing values for column based of a given strategy (mean, median, mode)

        args:

        column:  column with missing values to be filled
        strategy: how to fill in the missing values (mean, median, mode)
        """

        # fill in the missing values based on strategy 
        try :

            if strategy == "mean":
                col_mean = self.df[column].mean()
                print("column mean:",col_mean)
                self.df[column] = self.df[column].fillna(col_mean)
            elif strategy == "median":
                col_median = self.df[column].median()
                print("column median:", col_median)
                self.df[column] = self.df[column].fillna(col_median)
            elif strategy == "mode":
                col_mode = self.df[column].mode()
                print("col mode:", col_mode)
                self.df[column] = self.df[column].fillna(col_mode[0])
            else:
                print("strategy value not recognised use mean, median or mode")
        except ValueError as e :
            print(f"Error:  {e}")         

    def min_max_scale(self, column:str)-> None:
        """
        Applies min max scaling to the specifed column 

        args:

        column: the specified column to apply the scaling on 
        """
        
        try:
            # index the columns in  the dataframe
            col = self.df[column]
            #  min - max scaling of the columns in the dataframe
            self.df[column] = (col - col.min()) / (col.max() - col.min())
        except TypeError as e:
            print(f"Error: column must be numerical {e}")

    def encode_categorical(self, columns:list) -> None:
        """
        One hot encode a list of specified columns 

        args:

        columns : the specified columns to one hot encode
        """
        try:
            # one hot encode the given columns
            self.df = pd.get_dummies(self.df, columns=columns, drop_first=True, dtype=int)
        except Exception as e:
            print(f"error: column name not recognized {e}")

    def remove_columns(self, columns:list):
        """ 
        remove unneccessary  columns forn the data set

        args:

        columns: names of columns to be removed from the dataset 
        """
        # drop the columns
        self.df = self.df.drop(columns=columns)

    def get_clean_data(self) :
        """
        Return the clean dataset
        """
        # return the dataset
        return self.df 
    





