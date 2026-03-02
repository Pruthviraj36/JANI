import os
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# ── Load model bundle ────────────────────────────────────────────
bundle   = joblib.load('perfect_gpu_model.pkl')
model    = bundle['model']
encoders = bundle['encoders']
features = bundle['features']
params   = bundle.get('params', {})
print(f'✅ Model loaded. Features: {features}')

# ── In-memory prediction history ─────────────────────────────────
prediction_history = []

# ── Feature definitions (Visible in UI) ──────────────────────────
CATEGORICAL_OPTIONS = {
    'EmploymentType': ['Full-time', 'Part-time', 'Self-employed', 'Unemployed'],
    'LoanPurpose':    ['Home', 'Auto', 'Education', 'Business', 'Personal', 'Other'],
}

NUMERIC_FEATURES = [
    'Age', 'Income', 'LoanAmount', 'CreditScore',
    'MonthsEmployed', 'DTIRatio', 'InterestRate',
]

# ── Hidden Features (Sent to model with defaults) ────────────────
HIDDEN_DEFAULTS = {
    'Education':      "Bachelor's",
    'MaritalStatus':  "Married",
    'HasMortgage':    "No",
    'HasDependents':  "No",
    'HasCoSigner':    "No",
    'NumCreditLines': 5,
    'LoanTerm':       60,
}


# ── Routes ───────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', categorical_options=CATEGORICAL_OPTIONS)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        full_name = data.get('FullName', 'Unknown Borrower')

        # Build raw row for model
        row = {}
        # 1. Add visible numeric
        for feat in NUMERIC_FEATURES:
            row[feat] = float(data[feat])
        # 2. Add visible categorical
        for feat in CATEGORICAL_OPTIONS:
            row[feat] = data[feat]
        # 3. Add hidden defaults
        for feat, val in HIDDEN_DEFAULTS.items():
            row[feat] = val

        df = pd.DataFrame([row])

        # Apply LabelEncoders (uses the full encoders dict from bundle)
        for col, le in encoders.items():
            if col in df.columns:
                try:
                    df[col] = le.transform(df[col].astype(str))
                except ValueError:
                    df[col] = le.transform([le.classes_[0]])

        # Align column order to training (all 16 features)
        df = df[features]

        prob_high  = float(model.predict(df)[0])
        prob_low   = 1.0 - prob_high
        prediction = 1 if prob_high > 0.5 else 0

        result = {
            'prediction':     prediction,
            'label':          'High Risk' if prediction == 1 else 'Low Risk',
            'confidence':     round(max(prob_low, prob_high) * 100, 2),
            'prob_low_risk':  round(prob_low  * 100, 2),
            'prob_high_risk': round(prob_high * 100, 2),
        }

        # Store in history (including Full Name and Loan Type)
        history_entry = {
            **row,
            **result,
            'FullName': full_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'id': f'LN-{len(prediction_history)+1000:04d}',
        }
        prediction_history.append(history_entry)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/model-info')
def model_info():
    """Return model specifications and feature importance."""
    try:
        importance = model.feature_importance(importance_type='gain').tolist()
        total = sum(importance) or 1
        feature_importance = [
            {
                'feature': feat,
                'importance': round(imp, 2),
                'pct': round(imp / total * 100, 1),
            }
            for feat, imp in sorted(
                zip(features, importance),
                key=lambda x: x[1], reverse=True
            )
        ]

        return jsonify({
            'model_type': 'LightGBM (lgb.Booster)',
            'boosting_type': params.get('boosting_type', 'gbdt'),
            'objective': params.get('objective', 'binary'),
            'num_trees': model.num_trees(),
            'num_features': len(features),
            'features': features,
            'feature_importance': feature_importance,
            'params': {
                'learning_rate':   params.get('learning_rate'),
                'num_leaves':      params.get('num_leaves'),
                'max_depth':       params.get('max_depth'),
                'min_data_in_leaf':params.get('min_data_in_leaf'),
                'feature_fraction':params.get('feature_fraction'),
                'bagging_fraction':params.get('bagging_fraction'),
                'lambda_l1':       params.get('lambda_l1'),
                'lambda_l2':       params.get('lambda_l2'),
                'max_bin':         params.get('max_bin'),
            },
            'categorical_encoders': {
                col: le.classes_.tolist()
                for col, le in encoders.items()
            },
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history')
def history():
    """Return prediction history for Loan Portfolio page."""
    return jsonify(list(reversed(prediction_history)))


@app.route('/api/stats')
def stats():
    """Aggregate stats for dashboard KPIs."""
    total = len(prediction_history)
    high  = sum(1 for h in prediction_history if h['prediction'] == 1)
    avg_conf = (
        round(sum(h['confidence'] for h in prediction_history) / total, 1)
        if total else 0
    )
    return jsonify({
        'total': total,
        'high_risk': high,
        'low_risk': total - high,
        'default_rate': round(high / total * 100, 1) if total else 0,
        'avg_confidence': avg_conf,
    })


# ── Entry point ──────────────────────────────────────────────────
if __name__ == '__main__':
    port  = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
