from src.data_preprocessing import (
    load_data,
    inspect_data,
    split_features_target,
    create_preprocessor,
    split_data
)

from src.model_training import (
    create_model,
    create_param_grid,
    train_model,
    save_model
)

from src.evaluation import (
    predict,
    evaluate_model,
    print_metrics
)


DATA_PATH = "data_house.csv"


def main():
    # =========================
    # 1. DATA
    # =========================
    df = load_data(DATA_PATH)
    inspect_data(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)


    # =========================
    # 2. PREPROCESSING
    # =========================
    preprocessor = create_preprocessor(X_train)


    # =========================
    # 3. MODEL + GRID SEARCH
    # =========================
    pipeline = create_model(preprocessor)

    param_grid = create_param_grid()

    best_model = train_model(
        pipeline,
        param_grid,
        X_train,
        y_train
    )


    # =========================
    # 4. EVALUATION
    # =========================
    y_predict = predict(
        best_model,
        X_test
    )

    metrics = evaluate_model(
        y_test,
        y_predict
    )

    print_metrics(metrics)


    # =========================
    # 5. SAVE
    # =========================
    save_model(
        best_model,
        "models/best_model.pkl"
    )


if __name__ == "__main__":
    main()