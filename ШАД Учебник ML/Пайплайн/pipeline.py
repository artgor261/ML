from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
from typing import Optional, List

# Шкалировщик для численных признаков
class BaseDataPreprocessor(TransformerMixin):
    def __init__(self, needed_columns: Optional[List[str]]=None):
        self.needed_columns = needed_columns
        self.scaler = StandardScaler()

    def fit(self, data, *args):
        if self.needed_columns != None:
            self.scaler.fit(data[self.needed_columns].fillna(0))
        else:
            self.scaler.fit(data.fillna(0))
        return self

    def transform(self, data: pd.DataFrame) -> np.array:
        if self.needed_columns != None:
            return np.array(self.scaler.transform(data[self.needed_columns].fillna(0)))
        return np.array(self.scaler.transform(data.fillna(0)))


# Объединение энкодера и шкалировщика для слияние категориальных и численных признаков
class OneHotPreprocessor(BaseDataPreprocessor):
    def __init__(self, interesting_columns, **kwargs):
        super(OneHotPreprocessor, self).__init__(**kwargs)
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.interesting_columns = interesting_columns

    def fit(self, data, *args):
        self.encoder.fit(data[self.interesting_columns])
        super().fit(data)
        return self

    def transform(self, data):
        self.categorial_transformed = self.encoder.transform(data[self.interesting_columns]).toarray()
        self.scaler_transformed = super().transform(data)
        return np.hstack((self.scaler_transformed, self.categorial_transformed))


# Функция, в которой реализован пайплайн
def make_ultimate_pipeline(data_train, Y_train, data_test, interesting_columns, continuous_columns):
    grid_search = GridSearchCV(Ridge(), {"alpha":np.logspace(-3, 3, num=7, base=10.)}, cv=5)
    oneHot = OneHotPreprocessor(interesting_columns, needed_columns=continuous_columns)
    pipe = Pipeline([('oneHot', oneHot), ('grid_search', grid_search)])
    pipe.fit(data_train, Y_train)
    return pipe.predict(data_test)
