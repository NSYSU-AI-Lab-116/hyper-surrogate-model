import pandas as pd 
from sklearn.model_selection import train_test_split
from data import parser, preprocess, config
from models.ml_models import get_ml_models
from models.dl_models import MLPRegressor
from utils.trainer import train_dl_model
from utils.metrics import evaluate_predictions

def run_experiment(dataset_name="cifar10"):
    # read NAS-Bench-201
    df = parser.load_nasbench201()
    df = df[df["dataset"] == dataset_name]

    # 特徵處理
    X, y = preprocess.preprocess_dataframe(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ML baseline
    ml_models = get_ml_models()
    for name, model in ml_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"[ML] {name}:", evaluate_predictions(y_test, y_pred))

    # DL baseline
    dl_model = MLPRegressor(input_dim=X.shape[1])
    dl_results = train_dl_model(dl_model, X_train, y_train, X_test, y_test,
                                lr=config.LR, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS)
    print("[DL] MLP:", dl_results)

if __name__ == "__main__":
    for dataset in ["cifar10", "cifar100", "imagenet"]:
        print(f"\n=== Dataset: {dataset} ===")
        run_experiment(dataset)
