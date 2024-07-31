import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


class MostFrequentClassifier(ClassifierMixin):

    def fit(self, X=None, y=None):
        self.value = y.median()

    def predict(self, X=None):
        return np.full(shape=X.shape[0], fill_value=self.value)


data = pd.read_csv("C:/Users/Артем/Datasets/organisations.csv")
df = data[(data['average_bill'].notna()) & (data['average_bill'] <= 2500)]
clean_data = df
clean_data_train, clean_data_test = train_test_split(
    clean_data, stratify=clean_data['average_bill'],
    test_size=0.33, random_state=42)


clf = MostFrequentClassifier()
clf.fit(y=clean_data_train['average_bill'])

print("Balanced accuracy score: ",
      balanced_accuracy_score(clean_data_test['average_bill'],
                              clf.predict(clean_data_test)))
print("Accuracy score: ",
      accuracy_score(clean_data_test['average_bill'], clf.predict(clean_data_test)))

print("RMSE: ",
      np.sqrt(
          mean_squared_error(clean_data_test['average_bill'],
                             clf.predict(clean_data_test))))
