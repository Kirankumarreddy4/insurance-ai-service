from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

ROBOFLOW_API_KEY = "JiqpDWi01hEYtnMMa9bW"

MODEL_URL = "https://detect.roboflow.com/car-damage-detection-5ioys-iapbr/1"


@app.route("/")
def home():
    return "Vehicle Damage Detection API Running"


@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return jsonify({"success": False})

    image = request.files["image"]

    response = requests.post(
        MODEL_URL,
        params={"api_key": ROBOFLOW_API_KEY},
        files={"file": image}
    )

    result = response.json()

    return jsonify({
        "success": True,
        "count": len(result["predictions"]),
        "detections": result["predictions"]
    })


if __name__ == "__main__":
    app.run(debug=True)
