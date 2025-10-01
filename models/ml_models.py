from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def get_ml_models():
    return {
        "RandomForest": RandomForestRegressor(n_estimators=100),
        "SVM": SVR(kernel="rbf")
    }
