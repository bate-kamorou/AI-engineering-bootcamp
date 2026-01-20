from keras.models import load_model
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# load the test set X_test
X_test = pd.read_csv("data/processed/X_test.csv", index_col=None)

# load the true label set y_test
y_test = pd.read_csv("data/processed/y_test.csv", index_col=None)

# print(X_test.head(2))
# print(y_test.head(2))

# load the best model
try:
	best_model = load_model("models/best_titanic_nn5.keras")
	if best_model is None:
		raise ValueError("Model failed to load - file may not exist or be corrupted")
except Exception as e:
	print(f"Error loading model: {e}")
	raise

# make predictions with the model
y_preds = best_model.predict(X_test)

# convert the probablistic predictions into actual classes tresholding at 0.5

y_preds_nn = (y_preds > 0.5).astype("int32")

# confusion matrix
conf_mat = confusion_matrix(y_test, y_preds_nn)
print("confusion matrix:\n",conf_mat)

# classification report 
classi_report = classification_report(y_test, y_preds_nn)
print("classification report:\n",classi_report)

# classification report:
#                precision    recall  f1-score   support

#            0       0.82      0.90      0.86        78
#            1       0.84      0.73      0.78        56

#     accuracy                           0.83       134
#    macro avg       0.83      0.81      0.82       134
# weighted avg       0.83      0.83      0.83       134


# plot the model confusion matrix
conf_display  = ConfusionMatrixDisplay(conf_mat,)
conf_display.plot()
plt.show()