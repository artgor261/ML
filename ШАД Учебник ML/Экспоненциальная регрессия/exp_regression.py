import numpy as np
from sklearn.linear_model import Ridge


class ExponentialLinearRegression(Ridge):

    def fit(self, X, Y):
        super().fit(X, np.log(Y))
        return self

    def predict(self, X):
        return np.exp(super().predict(X))
