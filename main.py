import pandas as pd
from sklearn.model_selection import train_test_split
from data import parser, preprocess, config
from data.parser import load_dataset
from models.ml_models import get_ml_models
from models.dl_models import MLPRegressor
from utils.trainer import train_dl_model
from utils.metrics import evaluate_predictions

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor as SKMLPRegressor

dl_model = make_pipeline(
    StandardScaler(),
    SKMLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
    ),
)

svm = make_pipeline(
    StandardScaler(),
    SVR(kernel="rbf", C=10.0, gamma="scale")
)

def run_experiment(dataset: str):
    df = load_dataset(dataset=dataset)

    if df is None or df.empty:
        print(f"[ERROR] Loaded DataFrame is empty. Columns: {None if df is None else list(df.columns)}")
        return

    if "dataset" not in df.columns:
        print(f"[ERROR] 'dataset' column missing. Available columns: {list(df.columns)}")
        print(df.head())
        return

    # df = df[df["dataset"] == dataset].dropna(subset=["y"])
    if df.empty:
        print(f"[WARN] No data found for dataset: {dataset_name}")
        return

    # X, y = preprocess.preprocess_dataframe(df)
    X, y, feature_names = preprocess.preprocess_dataframe(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ml_models = get_ml_models()
    for name, model in ml_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"[ML] {name}:", evaluate_predictions(y_test, y_pred))

    dl_model = MLPRegressor(input_dim=X.shape[1])
    dl_results = train_dl_model(
        dl_model, X_train, y_train, X_test, y_test,
        lr=config.LR, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS
    )
    print("[DL] MLP:", dl_results)

if __name__ == "__main__":
    for dataset in ["cifar10", "cifar100", "imagenet16-120"]:
        print(f"\n- Dataset: {dataset}")
        run_experiment(dataset)
