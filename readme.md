# House Price Prediction

## 1. Giới thiệu

Dự án xây dựng mô hình **Machine Learning dự đoán giá nhà** dựa trên bộ dữ liệu California Housing được sử dụng dưới dạng file `data_house.csv`.

Bài toán thuộc nhóm **Supervised Learning → Regression**, trong đó mô hình học mối quan hệ giữa các đặc trưng của khu vực và giá nhà để dự đoán giá nhà cho dữ liệu mới.

Mô hình được sử dụng trong dự án:

* `RandomForestRegressor`
* `Pipeline`
* `ColumnTransformer`
* `SimpleImputer`
* `StandardScaler`
* `GridSearchCV`

---

## 2. Dataset

Dataset gồm các đặc trưng liên quan đến nhà ở và khu vực.

| Feature       | Ý nghĩa                                                             |
| ------------- | ------------------------------------------------------------------- |
| `MedInc`      | Median Income – thu nhập trung vị của các hộ gia đình trong khu vực |
| `HouseAge`    | Tuổi trung vị của các căn nhà trong khu vực                         |
| `AveRooms`    | Số phòng trung bình trên mỗi hộ gia đình                            |
| `AveBedrms`   | Số phòng ngủ trung bình trên mỗi hộ gia đình                        |
| `Population`  | Dân số trong khu vực                                                |
| `AveOccup`    | Số người trung bình trên mỗi hộ gia đình                            |
| `Latitude`    | Vĩ độ của khu vực                                                   |
| `Longitude`   | Kinh độ của khu vực                                                 |
| `MedHouseVal` | Giá nhà trung vị – **target cần dự đoán**                           |

### Input

```text
MedInc
HouseAge
AveRooms
AveBedrms
Population
AveOccup
Latitude
Longitude
```

### Target

```text
MedHouseVal
```

---

## 3. Bài toán

Cho thông tin về một khu vực:

```text
X = {
    MedInc,
    HouseAge,
    AveRooms,
    AveBedrms,
    Population,
    AveOccup,
    Latitude,
    Longitude
}
```

Mục tiêu là xây dựng mô hình:

```text
X → RandomForestRegressor → MedHouseVal
```

để dự đoán giá nhà.

---

## 4. Quy trình Machine Learning

```text
Dataset
   ↓
Load Data
   ↓
Train / Test Split
   ↓
Data Preprocessing
   │
   ├── SimpleImputer
   │
   └── StandardScaler
   ↓
ColumnTransformer
   ↓
Pipeline
   ↓
RandomForestRegressor
   ↓
GridSearchCV
   ↓
Best Model
   ↓
Prediction
   ↓
Evaluation
```

---

## 5. Data Preprocessing

### 5.1. Xử lý Missing Value

Sử dụng:

```python
SimpleImputer(strategy="median")
```

Các giá trị bị thiếu sẽ được thay thế bằng **median của feature tương ứng**.

### 5.2. Chuẩn hóa dữ liệu

Sử dụng:

```python
StandardScaler()
```

Các feature numerical được chuẩn hóa trước khi đưa vào pipeline.

### 5.3. ColumnTransformer

Sử dụng `ColumnTransformer` để xác định các feature numerical cần áp dụng preprocessing.

Hiện tại dataset chỉ sử dụng các feature dạng số nên các feature được xử lý cùng một pipeline:

```text
Numerical Features
       ↓
SimpleImputer
       ↓
StandardScaler
```

---

## 6. Model

Sử dụng:

```python
RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)
```

Random Forest là mô hình ensemble kết hợp nhiều Decision Tree để thực hiện dự đoán.

Trong bài toán này, Random Forest nhận các feature sau preprocessing và dự đoán `MedHouseVal`.

---

## 7. Hyperparameter Tuning

Sử dụng `GridSearchCV` để tìm tổ hợp hyperparameter tốt nhất cho Random Forest.

Các hyperparameter được tìm kiếm:

```python
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__criterion": [
        "squared_error",
        "absolute_error",
        "poisson"
    ]
}
```

Sử dụng:

```python
cv=4
```

và metric:

```python
scoring="r2"
```

GridSearchCV sẽ thử các tổ hợp hyperparameter và lựa chọn mô hình có kết quả tốt nhất theo R² trên cross-validation.

---

## 8. Evaluation

Sau khi tìm được model tốt nhất, mô hình được sử dụng để dự đoán trên `X_test`.

Các metric được sử dụng:

### R² – R-squared

Đánh giá mức độ mô hình giải thích được sự biến thiên của target.

**R² càng cao càng tốt.**

### MAE – Mean Absolute Error

Đo sai số tuyệt đối trung bình giữa giá trị thực tế và giá trị dự đoán.

**MAE càng thấp càng tốt.**

### MSE – Mean Squared Error

Đo trung bình bình phương sai số.

**MSE càng thấp càng tốt.**

### RMSE – Root Mean Squared Error

Căn bậc hai của MSE.

**RMSE càng thấp càng tốt.**

---

## 9. Kết quả cần thu được

Sau khi chạy chương trình, cần ghi nhận:

```text
Best Parameters
R²
MAE
MSE
RMSE
```

## 10. Cấu trúc Project

```text
house_price_prediction/
│
├── data/
│   └── data_house.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
│
├── models/
│   └── best_model.pkl
│
├── reports/
│   └── results.csv
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 11. Công nghệ sử dụng

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

Các thư viện chính:

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
```

---

## 12. Mục tiêu cuối cùng

Xây dựng một pipeline Machine Learning hoàn chỉnh:

```text
Raw Data
   ↓
Preprocessing
   ↓
Random Forest
   ↓
GridSearchCV
   ↓
Best Model
   ↓
House Price Prediction
   ↓
Evaluation
```

Mô hình cuối cùng có thể được lưu lại để sử dụng cho việc dự đoán giá nhà trên dữ liệu mới.
