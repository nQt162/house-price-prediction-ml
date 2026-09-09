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
    Đọc CSV và trả về DataFrame
    """
    df = pd.read_csv(path)
    return df


def inspect_data(df):
    """
    Khám phá dữ liệu: shape, info, corr, missing, duplicate, describe
    """
    print("=" * 40)
    print("SHAPE:", df.shape)
    print("=" * 40)

    df.info()

    print("=" * 40)
    print("MISSING VALUES:")
    print(df.isnull().sum())

    print("=" * 40)
    print("DUPLICATES:", df.duplicated().sum())

    print("=" * 40)
    print("DESCRIBE:")
    print(df.describe())

    print("=" * 40)
    print("CORRELATION:")
    print(df.corr(numeric_only=True))

    print("=" * 40)


def split_features_target(df):
    """
    X = toàn bộ feature, y = TARGET
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y


def create_preprocessor(x):
    """
    Pipeline: SimpleImputer(median) + StandardScaler
    bọc trong ColumnTransformer cho numerical features
    """
    num_features = list(
        x.select_dtypes(include="number").columns
    )

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("num_feature", num_transformer, num_features)
    ])

    return preprocessor


def split_data(X, y):
    """
    train_test_split với test_size=0.2, random_state=42
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test