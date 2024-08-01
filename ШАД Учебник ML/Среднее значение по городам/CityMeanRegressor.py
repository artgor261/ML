import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class CityMeanRegressor(RegressorMixin):
    def fit(self, X=None, y=None):
        self.msk_mean = y[y["city"] == "msk"]["average_bill"].mean()
        self.spb_mean = y[y["city"] == "spb"]["average_bill"].mean()

    def predict(self, X=None):
        self.array = list()

        for index, row in X.iterrows():
            if row["city"] == "msk":
                self.array.append(self.msk_mean)
            else:
                self.array.append(self.spb_mean)

        return self.array


reg = CityMeanRegressor()
reg.fit(y=clean_data_train)
city_mean_rmse = np.sqrt(mean_squared_error(clean_data_test["average_bill"], reg.predict(clean_data_test)))
s = pd.Series(np.hstack([base_rmse, city_mean_rmse]), index=["BaseLine RMSE", "CityMeanRegressor RMSE"])
print(s)
