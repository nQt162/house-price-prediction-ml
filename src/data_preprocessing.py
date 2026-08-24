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
    df = pd.read_csv(path)
    return df 


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
    
    print("Shape:")
    print(df.shape)

    print("\nInfo:")
    print(df.info())

    print("\nCorrelation:")
    print(df.corr(numeric_only=True))

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicates:")
    print(df.duplicated().sum())

    print("\nDescription:")
    print(df.describe())
   


def split_features_target(df):
    """
    TODO:
    x = toàn bộ feature
    y = TARGET
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y
   


def create_preprocessor(x):
    """
    TODO:
    - Xác định numerical features
    - Tạo Pipeline:
        SimpleImputer
        StandardScaler
    - Tạo ColumnTransformer
    """
    # 1. Lấy các cột dữ liệu số
    numerical_features = x.select_dtypes(include=["int64", "float64"]).columns

    # 2. Tạo pipeline cho dữ liệu số
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 3. Tạo ColumnTransformer
    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features)
    ])

    return preprocessor


def split_data(X, y):
    """
    TODO:
    train_test_split
    test_size = 0.2
    random_state = 42
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=4
    )

    return X_train, X_test, y_train, y_test