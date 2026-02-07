from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import joblib

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "machine-learning"))
from preprocessing import apply_artifacts_to_input


MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "machine-learning", "models")


def load_model_bundle():
    model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))
    artifacts = joblib.load(os.path.join(MODELS_DIR, "preprocess_artifacts.joblib"))
    features_path = os.path.join(MODELS_DIR, "features.json")
    features = artifacts.get("features", [])
    if os.path.exists(features_path):
        with open(features_path, "r", encoding="utf-8") as f:
            features = json.load(f)

    threshold = 0.5
    thr_path = os.path.join(MODELS_DIR, "risk_threshold.json")
    if os.path.exists(thr_path):
        try:
            with open(thr_path, "r", encoding="utf-8") as f:
                threshold = float(json.load(f).get("threshold", 0.5))
        except Exception:
            threshold = 0.5
    return model, artifacts, features, threshold


app = Flask(__name__)
CORS(app)
model, artifacts, features, decision_threshold = load_model_bundle()

def generate_recommendations(payload: dict, proba: float) -> dict:
    lifestyle = []
    medical = []
    if payload.get("tobacco", 0) == 1:
        lifestyle.append("Berhenti merokok; minta bantuan program berhenti merokok.")
    if payload.get("alcohol", 0) == 1:
        lifestyle.append("Kurangi/stop konsumsi alkohol.")
    if payload.get("exercise", 0) == 0:
        lifestyle.append("Rutin olahraga aerobik sedang 150 menit/minggu.")
    imc = payload.get("imc")
    if isinstance(imc, (int, float)) and imc is not None:
        if imc >= 30:
            lifestyle.append("Fokus penurunan berat badan bertahap (diet seimbang + aktivitas fisik).")
        elif imc >= 25:
            lifestyle.append("Pertahankan IMC sehat dengan pola makan seimbang.")
    if payload.get("breastfeeding", 0) == 0 and payload.get("children", 0) > 0:
        lifestyle.append("Menyusui di masa depan dapat menurunkan risiko (bila relevan).")

    if proba >= 0.5:
        medical.append("Konsultasi dokter untuk evaluasi risiko dan skrining lebih lanjut.")
        medical.append("Pertimbangkan mammografi/USG payudara sesuai usia & pedoman klinis.")
    elif proba >= 0.35:
        medical.append("Diskusikan dengan dokter mengenai interval skrining yang sesuai.")
    else:
        medical.append("Lanjutkan pola hidup sehat dan skrining rutin sesuai usia.")
    if payload.get("nrelbc", 0) and payload.get("nrelbc", 0) >= 1:
        medical.append("Riwayat keluarga terdeteksi: pertimbangkan konseling genetik.")

    return {"lifestyle": lifestyle, "medical": medical}


@app.route("/", methods=["GET"])
def index() -> tuple:
    return jsonify({
        "service": "BreastGuard API",
        "status": "ok",
        "endpoints": {
            "health": {"method": "GET", "path": "/health"},
            "predict": {"method": "POST", "path": "/predict"}
        }
    }), 200


@app.route("/health", methods=["GET"])
def health() -> tuple:
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict() -> tuple:
    payload = request.get_json(force=True, silent=True) or {}
    try:
        X = apply_artifacts_to_input(payload, artifacts)
        probs = model.predict_proba(X.values)[:, 1]
        proba = float(probs[0])
        thr_high = request.args.get("threshold_high")
        thr_med = request.args.get("threshold_medium")
        try:
            high_threshold = float(thr_high) if thr_high is not None else float(decision_threshold)
        except Exception:
            high_threshold = float(decision_threshold)
        try:
            medium_threshold = float(thr_med) if thr_med is not None else 0.35
        except Exception:
            medium_threshold = 0.35
        if proba >= high_threshold:
            risk_category = "high"
        elif proba >= medium_threshold:
            risk_category = "medium"
        else:
            risk_category = "low"
        pred = 1 if risk_category in ("medium", "high") else 0
        risk_percent = round(proba * 100.0, 2)
        # Generate recommendations based on input and risk level
        recommendations = generate_recommendations(payload, proba)
        
        if request.args.get("debug", "false").lower() == "true":
            processed = {col: float(X.iloc[0][col]) if col in X.columns else None for col in X.columns}
            return jsonify({
                "pred": pred,
                "proba": proba,
                "risk_percent": risk_percent,
                "risk_category": risk_category,
                "thresholds": {"high": high_threshold, "medium": medium_threshold},
                "recommendations": recommendations,
                "processed": processed
            }), 200
        return jsonify({
            "pred": pred,
            "proba": proba,
            "risk_percent": risk_percent,
            "risk_category": risk_category,
            "thresholds": {"high": high_threshold, "medium": medium_threshold},
            "recommendations": recommendations
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)