from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Optional, List
import sklearn.base

class BaseDataPreprocessor(TransformerMixin):
    def __init__(self, needed_columns: Optional[List[str]]=None):
        self.needed_columns = needed_columns
        self.scaler = StandardScaler()

    def fit(self, data, *args):
        if self.needed_columns != None:
            self.scaler.fit(data[self.needed_columns])
        else:
            self.scaler.fit(data)
        self.mean = self.scaler.mean_
        return self

    def transform(self, data: pd.DataFrame) -> np.array:
        if self.needed_columns != None:
            return np.array(self.scaler.transform(data[self.needed_columns]))
        return np.array(self.scaler.transform(data))
