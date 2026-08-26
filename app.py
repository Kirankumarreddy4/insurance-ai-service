from flask import Flask, request, jsonify
import requests
import base64
import tempfile
import os
import cv2
import numpy as np

app = Flask(__name__)

ROBOFLOW_API_KEY = "JiqpDWi01hEYtnMMa9bW"

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

        ##################################################
        # Decode Base64 Image
        ##################################################

        image_bytes = base64.b64decode(data["image"])

        image_np = np.frombuffer(image_bytes, np.uint8)

        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        suffix = ".jpg"

        if "fileName" in data:
            _, suffix = os.path.splitext(data["fileName"])

        ##################################################
        # Save temp image for Roboflow
        ##################################################

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

        result = response.json()

        ##################################################
        # Draw Bounding Boxes
        ##################################################

        predictions = result.get("predictions", [])

        for pred in predictions:

            x = pred["x"]
            y = pred["y"]
            w = pred["width"]
            h = pred["height"]

            cls = pred["class"]
            conf = pred["confidence"]

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)

            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            color = (0,255,0)

            cv2.rectangle(
                image,
                (x1,y1),
                (x2,y2),
                color,
                3
            )

            label = f"{cls} ({conf:.2f})"

            cv2.putText(
                image,
                label,
                (x1,max(30,y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        ##################################################
        # Convert annotated image to Base64
        ##################################################

        _, buffer = cv2.imencode(".jpg", image)

        annotated_base64 = base64.b64encode(buffer).decode("utf-8")

        ##################################################
        # Return everything
        ##################################################

        return jsonify({

            "success": True,

            "predictionCount": len(predictions),

            "predictions": predictions,

            "annotatedImage": annotated_base64,

            "image": result.get("image"),

            "inference_id": result.get("inference_id"),

            "time": result.get("time")

        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


if __name__ == "__main__":

    port = int(os.environ.get("PORT",5000))

    app.run(host="0.0.0.0", port=port)
