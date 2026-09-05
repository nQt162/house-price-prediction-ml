'''
Nhiệm vụ:
  Predict test set
  Tính metric
  Phân tích kết quả
  Có thể vẽ biểu đồ
'''
import numpy as np
import pandas as pd
from sklearn.metrics import  r2_score,  mean_absolute_error, mean_squared_error


def predict(model, X_test):
    """
    Dự đoán trên tập test
    """
    y_predict = model.predict(X_test)
    return y_predict


def evaluate_model(y_test, y_predict):
    """
    Tính R², MAE, MSE, RMSE — trả về dictionary
    """
    mse = mean_squared_error(y_test, y_predict)

    metrics = {
        "R2": r2_score(y_test, y_predict),
        "MAE": mean_absolute_error(y_test, y_predict),
        "MSE": mse,
        "RMSE": np.sqrt(mse)
    }

    return metrics


def print_metrics(metrics):
    """
    In kết quả đẹp
    """
    print("=" * 40)
    print("KẾT QUẢ ĐÁNH GIÁ MODEL")
    print("=" * 40)
    for name, value in metrics.items():
        print(f"{name:<6}: {value:.4f}")
    print("=" * 40)


def save_results(metrics, path):
    """
    Lưu kết quả vào CSV
    """
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(path, index=False)
    print("Kết quả đã lưu tại:", path)