from flask import Flask, request, jsonify
import requests
import base64
import tempfile
import os

app = Flask(__name__)

ROBOFLOW_API_KEY = "JiqpDWi01hEYtnMMa9bW"

MODEL_URL = "https://detect.roboflow.com/car-damage-detection-5ioys-iapbr/1"


@app.route("/")
def home():
    return "Vehicle Damage Detection API Running"

app = Flask(__name__)

ROBOFLOW_API_KEY = "YOUR_API_KEY"

MODEL_URL = "https://detect.roboflow.com/car-damage-detection-5ioys-iapbr/1"


@app.route("/")
def home():
    return "Vehicle Damage Detection API Running"


@app.route("/detect", methods=["POST"])
def detect():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "No JSON body received"
            }), 400

        if "image" not in data:
            return jsonify({
                "success": False,
                "message": "Image not provided"
            }), 400

        image_bytes = base64.b64decode(data["image"])

        suffix = ".jpg"

        if "fileName" in data:
            _, suffix = os.path.splitext(data["fileName"])

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:

            temp.write(image_bytes)
            temp.flush()

            with open(temp.name, "rb") as img:

                response = requests.post(
                    MODEL_URL,
                    params={
                        "api_key": ROBOFLOW_API_KEY
                    },
                    files={
                        "file": img
                    }
                )

        os.remove(temp.name)

        return jsonify(response.json())

    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
