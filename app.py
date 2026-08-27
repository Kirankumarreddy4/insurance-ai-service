from flask import Flask, request, jsonify
import requests
import base64
import tempfile
import os
import cv2
import numpy as np
import traceback
import hashlib
from google import genai
import json
import re
from google.genai import types
import time

GEMINI_API_KEY = "AQ.Ab8RN6Jh0ibqvZ2n-Lh9iYiaTvWddSU9I0bW3LYSzFdoVDQC6Q"
ROBOFLOW_API_KEY = "JiqpDWi01hEYtnMMa9bW"
MODEL_URL = "https://detect.roboflow.com/car-damage-detection-5ioys-iapbr/1"

client = genai.Client(api_key=GEMINI_API_KEY)
app = Flask(__name__)


@app.route("/")
def home():
    return "Vehicle Damage Detection API Running"


# ################################################
# Generate a unique color for each class
# ################################################
def get_color(class_name):
    digest = hashlib.md5(class_name.encode()).digest()

    b = digest[0]
    g = digest[1]
    r = digest[2]

    # Brighten the colors
    b = max(80, b)
    g = max(80, g)
    r = max(80, r)

    return (int(b), int(g), int(r))


def estimate_damage_with_gemini(
    annotated_base64,
    predictions,
    vehicle,
    claim,
    original_image_base64
):
    prompt = f"""
You are a senior automobile insurance surveyor.

Analyse BOTH the vehicle image and Roboflow detections.

Vehicle
{json.dumps(vehicle, indent=2)}

Claim
{json.dumps(claim, indent=2)}

Detected damages
{json.dumps(predictions, indent=2)}

Instructions:
- Use BOTH the image and detections.
- Do not estimate based only on detections.
- If multiple damages belong to same panel, estimate realistic repair.
- If bumper is cracked, recommend replacement.
- If dent is repairable, recommend repair.

Return ONLY valid JSON in this format:
{{
    "severity": "",
    "estimatedCostMin": 0,
    "estimatedCostMax": 0,
    "currency": "INR",
    "laborHours": 0,
    "partsToReplace": [],
    "partsToRepair": [],
    "confidence": 0,
    "recommendation": "",
    "summary": ""
}}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            prompt,
            types.Part.from_bytes(
                data=base64.b64decode(original_image_base64),
                mime_type="image/jpeg"
            )
        ]
    )

    text = response.text
    match = re.search(r"{.*}", text, re.DOTALL)

    if match:
        text = match.group(0)

    return json.loads(text)


@app.route("/detect", methods=["POST"])
def detect():
    start = time.time()
    temp_path = None

    try:
        # ################################################
        # Read Request
        # ################################################
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

        vehicle = data.get("vehicle", {})
        claim = data.get("claim", {})
        original_base64 = data["image"]

        # ################################################
        # Decode Image
        # ################################################
        image_bytes = base64.b64decode(data["image"])
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                "success": False,
                "message": "Unable to decode image"
            }), 400

        # ################################################
        # File Extension
        # ################################################
        suffix = ".jpg"
        if "fileName" in data:
            _, suffix = os.path.splitext(data["fileName"])

        # ################################################
        # Save Temp Image
        # ################################################
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(image_bytes)
            temp.flush()
            temp_path = temp.name

        # ################################################
        # Call Roboflow
        # ################################################
        with open(temp_path, "rb") as img:
            response = requests.post(
                MODEL_URL,
                params={
                    "api_key": ROBOFLOW_API_KEY
                },
                files={
                    "file": img
                },
                timeout=60
            )

        # ################################################
        # Delete Temp File
        # ################################################
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            temp_path = None

        # ################################################
        # Validate Response
        # ################################################
        if response.status_code != 200:
            return jsonify({
                "success": False,
                "roboflowStatus": response.status_code,
                "response": response.text
            }), response.status_code

        result = response.json()

        predictions = result.get("predictions", [])
        predictions = [
            p for p in predictions
            if p["confidence"] >= 0.35
        ]

        detectedParts = list(
            set(
                p["class"]
                for p in predictions
            )
        )

        avgConfidence = 0
        if predictions:
            avgConfidence = round(
                sum(
                    p["confidence"]
                    for p in predictions
                ) / len(predictions),
                2
            )

        # ################################################
        # No Damages
        # ################################################
        if len(predictions) == 0:
            return jsonify({
                "success": True,
                "predictionCount": 0,
                "predictions": [],
                "image": result.get("image"),
                "inference_id": result.get("inference_id"),
                "time": result.get("time")
            })

        # ################################################
        # Draw Bounding Boxes
        # ################################################
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

            color = get_color(cls)

            # #########################################
            # Bounding Box
            # #########################################
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            # #########################################
            # Label
            # #########################################
            label = f"{cls} ({conf:.2f})"

            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )

            label_y = max(text_height + 8, y1 - 8)

            cv2.rectangle(
                image,
                (x1, label_y - text_height - 8),
                (x1 + text_width + 8, label_y + baseline),
                color,
                -1
            )

            cv2.putText(
                image,
                label,
                (x1 + 4, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # ################################################
        # Compress Image
        # ################################################
        encode_param = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            65
        ]

        success, buffer = cv2.imencode(
            ".jpg",
            image,
            encode_param
        )

        if not success:
            return jsonify({
                "success": False,
                "message": "Unable to encode annotated image"
            }), 500

        annotated_base64 = base64.b64encode(buffer).decode("utf-8")

        # ################################################
        # Estimate Damage with Gemini
        # ################################################
        assessment = None
        elapsed = 0

        try:
            assessment = estimate_damage_with_gemini(
                annotated_base64,
                predictions,
                vehicle,
                claim,
                original_base64
            )
            elapsed = round(time.time() - start, 2)

        except Exception as ex:
            print("Gemini Error:", ex)
            assessment = {
                "severity": "Unknown",
                "estimatedCostMin": 0,
                "estimatedCostMax": 0,
                "currency": "INR",
                "laborHours": 0,
                "partsToReplace": [],
                "partsToRepair": [],
                "confidence": 0,
                "recommendation": "Unable to estimate",
                "summary": str(ex)
            }

        # ################################################
        # Response
        # ################################################
        return jsonify({
            "success": True,
            "predictionCount": len(predictions),
            "predictions": predictions,
            "assessment": assessment,
            "annotatedImage": annotated_base64,
            "image": result.get("image"),
            "inference_id": result.get("inference_id"),
            "time": result.get("time"),
            "geminiTime": elapsed,
            "modelInfo": {
                "detector": "Roboflow v1",
                "estimator": "Gemini 3.6 Flash"
            },
            "detectedParts": detectedParts,
            "averageConfidence": avgConfidence,
        })

    except Exception as e:
        traceback.print_exc()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port
    )
