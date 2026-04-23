import os
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from models import run_all

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# FILE READER
# =========================
def read_file(file, filename):
    filename = filename.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(file)
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format")


# =========================
# DATA VALIDATION
# =========================
def validate_dataset(df):
    errors = []

    required_cols = ["Potential", "OXIDATION", "Zn/Co_Conc", "SCAN_RATE", "ZN", "CO", "Current"]

    # column check
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")

    if errors:
        return errors

    # empty check
    if df.empty:
        errors.append("Dataset is empty")

    # missing values
    if df.isnull().sum().sum() > 0:
        errors.append("Dataset contains missing values")

    # scientific sanity checks
    if (df["SCAN_RATE"] <= 0).any():
        errors.append("Scan rate must be positive")

    if (df["Zn/Co_Conc"] < 0).any():
        errors.append("Concentration cannot be negative")

    if df["Current"].abs().max() > 1e6:
        errors.append("Current values unrealistic")

    return errors


# =========================
# RESULT VALIDATION
# =========================
def validate_results(result):
    warnings = []

    try:
        best = result["models"][result["best_model"]]["optimization"]

        if best["capacitance"] <= 0:
            warnings.append("Capacitance is non-physical")

        if best["energy"] <= 0:
            warnings.append("Energy density invalid")

        if best["power"] <= 0:
            warnings.append("Power density invalid")

    except:
        warnings.append("Result validation failed")

    return warnings


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        train_file = request.files.get("train")
        test_file = request.files.get("test")

        train_name = request.form.get("train_name")
        test_name = request.form.get("test_name")

        # =====================
        # FILE HANDLING
        # =====================
        if train_file and test_file:
            train_path = os.path.join(UPLOAD_FOLDER, str(int(time.time())) + "_" + train_file.filename)
            test_path = os.path.join(UPLOAD_FOLDER, str(int(time.time())) + "_" + test_file.filename)

            train_file.save(train_path)
            test_file.save(test_path)

        elif train_name and test_name:
            train_path = os.path.join(UPLOAD_FOLDER, train_name)
            test_path = os.path.join(UPLOAD_FOLDER, test_name)

        else:
            return jsonify({"error": "Upload OR select both files"}), 400

        # =====================
        # READ FILES
        # =====================
        train_df = read_file(open(train_path, "rb"), train_path)
        test_df = read_file(open(test_path, "rb"), test_path)

        # =====================
        # VALIDATION
        # =====================
        train_errors = validate_dataset(train_df)
        test_errors = validate_dataset(test_df)

        if train_errors or test_errors:
            return jsonify({
                "error": "Dataset validation failed",
                "train_errors": train_errors,
                "test_errors": test_errors
            }), 400

        # =====================
        # RUN MODEL
        # =====================
        result = run_all(train_df, test_df)

        if "error" in result:
            return jsonify(result), 500

        # =====================
        # RESULT VALIDATION
        # =====================
        warnings = validate_results(result)
        result["warnings"] = warnings

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/files", methods=["GET"])
def list_files():
    return jsonify({"files": os.listdir(UPLOAD_FOLDER)})


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))