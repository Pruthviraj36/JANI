import joblib
import pickle

filename = 'models/loan_risk_model.pkl'

# Try joblib first (most common for sklearn models)
try:
    model = joblib.load(filename)
    print("Loaded with joblib!")
except Exception as e:
    print(f"joblib failed: {e}")
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print("Loaded with pickle!")
    except Exception as e2:
        print(f"pickle also failed: {e2}")
        exit()

print("Type:", type(model))
if hasattr(model, 'feature_names_in_'):
    print("Features:", list(model.feature_names_in_))
elif hasattr(model, 'n_features_in_'):
    print("Num features:", model.n_features_in_)
if hasattr(model, 'classes_'):
    print("Classes:", list(model.classes_))
if hasattr(model, 'get_params'):
    print("Params:", model.get_params())
