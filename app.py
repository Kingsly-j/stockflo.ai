from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# =========================
# LOAD MODEL + COLUMNS
# =========================
model = joblib.load("stockflow_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# =========================
# HOME
# =========================
@app.route("/")
def home():
    return "StockFlow AI is running 🚀"

# =========================
# PREDICT
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    # Build the model row in one pass. Repeatedly adding 1,000+ DataFrame
    # columns per request is slow and fragments Pandas memory.
    row = {col: data.get(col, 0) for col in model_columns}
    df = pd.DataFrame([row], columns=model_columns)

    # Predict
    prediction = model.predict(df)[0]

    return jsonify({
        "prediction": float(prediction)
    })

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
