import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from collections import Counter


class MostFrequentClassifier(ClassifierMixin):

    def fit(self, X=None, y=None):
        self.msk_median = (y[y["city"] == "msk"][["average_bill", "modified_rubrics"]]
                           .groupby(["modified_rubrics"]).median())
        self.spb_median = (y[y["city"] == "spb"][["average_bill", "modified_rubrics"]]
                           .groupby(["modified_rubrics"]).median())

    def predict(self, X=None):
        self.arr = list()
        for index, row in X.iterrows():
            if row["city"] == "msk":
                self.arr.append(self.msk_median.loc[row["modified_rubrics"]]["average_bill"])
            else:
                self.arr.append(self.spb_median.loc[row["modified_rubrics"]]["average_bill"])
        return self.arr


def insert_column(df):
    arr = list()
    cnt = Counter(df["rubrics_id"])

    for index, row in df.iterrows():
        if cnt[row["rubrics_id"]] > 100:
            arr.append(row["rubrics_id"])
        else:
            arr.append("other")
    df.insert(len(df.columns), "modified_rubrics", arr)


data = pd.read_csv("C:/Users/Артем/Datasets/organisations.csv")
clean_data = data[(data['average_bill'].notna()) & (data['average_bill'] <= 2500)]
insert_column(clean_data)
clean_data_train, clean_data_test = train_test_split(
    clean_data, stratify=clean_data['average_bill'],
    test_size=0.33, random_state=42)

clf = MostFrequentClassifier()
clf.fit(y=clean_data_train)

print("RMSE: ", np.sqrt(mean_squared_error(clean_data_test["average_bill"],
                                           clf.predict(clean_data_test))))

print("Accuracy score: ", accuracy_score(clean_data_test["average_bill"],
                                         clf.predict(clean_data_test)))

print("Balanced accuracy score: ", balanced_accuracy_score(clean_data_test["average_bill"],
                                                           clf.predict(clean_data_test)))
