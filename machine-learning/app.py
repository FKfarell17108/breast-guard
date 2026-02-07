from flask import Flask, request, jsonify
from inference import BreastCancerRiskModel


app = Flask(__name__)
model = BreastCancerRiskModel()


@app.route("/", methods=["GET"])
def index() -> tuple:
    return jsonify({
        "service": "BreastGuard API",
        "status": "ok",
        "endpoints": {
            "health": {"method": "GET", "path": "/health"},
            "predict": {"method": "POST", "path": "/predict", "body": {
                "age": 45, "menarche": 13, "menopause": 1, "agefirst": 22,
                "children": 2, "breastfeeding": 1, "nrelbc": 0, "imc": 23.5,
                "weight": 60, "exercise": 1, "alcohol": 0, "tobacco": 0, "allergies": 0
            }}
        }
    }), 200


@app.route("/health", methods=["GET"])
def health() -> tuple:
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict() -> tuple:
    payload = request.get_json(force=True, silent=True) or {}
    try:
        result = model.predict_risk(payload)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


