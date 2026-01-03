from Day_6_data_cleaner import DataCleaner
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot  as plt
import joblib
import os 



# run the full pipeline from data ingestion to model prediction
def run_full_pipeline(data_path:str): 
    # ______________ load the data ____________

    data = DataCleaner(data_path)

    df = data.clean_all("Age", "median", "Fare", ["Sex", "Embarked"], ["Name", "Ticket", "Cabin","PassengerId"])

    #_____________ Features and labesl ____________

    y = df["Survived"]
    X = df.drop(columns="Survived")

    # ______________________ Split into train and test sets __________
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.25)

    # ____________________  Build and fit the model ______

    # initialize the model with the best hyperparameters for GridsearchCv
    cls = RandomForestClassifier(n_estimators=300,max_depth=5, min_samples_split=3)

    # fit the model on the training set
    cls.fit(X_train,y_train)
    
    # _____________ Evaluate ____________

    # print the model score
    print("model accuracy score is : ", cls.score(X_test, y_test))

    # model predictions'
    y_preds = cls.predict(X_test)

    # print plot the classification report 

    print("classificaation report : \n", classification_report(y_test, y_preds))

    # plot the confusion matrix
    conf_mat = confusion_matrix(y_test, y_preds)
    conf_mat_display = ConfusionMatrixDisplay(conf_mat)
    conf_mat_display.plot()
    plt.title("Confusion matrix")
    plt.show()  

    #_______________ save the model _____________
    if not os.path.exists("models"):
        os.makedirs("models")
        joblib.dump(value=cls, filename="models\\titanic_final_model.joblib")
        print("Path creates and Titanic final model saved")
        
    else :
        joblib.dump(value=cls, filename="models\\titanic_final_model.joblib")
        print("Titanic final model saved")

    return cls



    
    
 
# run the run_full_pipeline function 

if __name__ == "__main__" :
    run_full_pipeline("data\\raw\\titanic.csv")