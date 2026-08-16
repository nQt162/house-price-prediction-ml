'''
Nhiệm vụ:
  Load dataset 
  Hiểu dữ liệu
  Kiểm tra dữ liệu
  Xác định X, y
  Xác định numerical/categorical features
  Xây dựng ColumnTransformer
  Xây dựng SimpleImputer
  StandardScaler
  Train/test split
'''

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


TARGET = "MedHouseVal"


def load_data(path):
    """
    TODO:
    - Đọc CSV
    - Trả về DataFrame
    """
    pass


def inspect_data(df):
    """
    TODO:
    - df.shape
    - df.info()
    - df.corr()
    - missing values
    - duplicate
    - describe()
    """
    pass


def split_features_target(df):
    """
    TODO:
    x = toàn bộ feature
    y = TARGET
    """
    pass


def create_preprocessor(x):
    """
    TODO:
    - Xác định numerical features
    - Tạo Pipeline:
        SimpleImputer
        StandardScaler
    - Tạo ColumnTransformer
    """
    pass


def split_data(X, y):
    """
    TODO:
    train_test_split
    test_size = 0.2
    random_state = 42
    """
    pass