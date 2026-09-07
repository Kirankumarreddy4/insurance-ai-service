from flask import Flask, request, jsonify
import base64
import hashlib
import json
import os
import re
import tempfile
import time
import traceback

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

for model in client.models.list():
    print(model.name)

MODEL_URL = (
    "https://detect.roboflow.com/"
    "car-damage-detection-5ioys-iapbr/1"
)

app = Flask(__name__)


@app.route("/")
def home():
    return "Vehicle Damage Detection API Running"


# --------------------------------------------------
# Generate Unique Color for Each Class
# --------------------------------------------------

def get_color(class_name):
    digest = hashlib.md5(class_name.encode()).digest()

    b = max(80, digest[0])
    g = max(80, digest[1])
    r = max(80, digest[2])

    return int(b), int(g), int(r)


# --------------------------------------------------
# Gemini Damage Estimation
# --------------------------------------------------

def estimate_damage_with_gemini(
    annotated_base64,
    predictions,
    vehicle,
    claim,
    original_image_base64,
):
    schema = {
        "damageDetected": True,
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
    }

    prompt = f"""
    You are an experienced automobile insurance surveyor.

    Analyse the ORIGINAL vehicle image.

    If Roboflow detections are available, use them as supporting evidence.

    If no detections are available, perform your own visual assessment from the image.

    Vehicle
    {json.dumps(vehicle, indent=2)}

    Claim
    {json.dumps(claim, indent=2)}

    Roboflow detections
    {json.dumps(predictions, indent=2)}

    Instructions

    - Carefully inspect the image.
    - Do NOT rely only on Roboflow.
    - If Roboflow missed visible damage, identify it yourself.
    - If the image appears undamaged, clearly state that.
    - Estimate realistic repair costs.
    - Recommend repair or replacement where appropriate.
    - If image quality is insufficient, reduce confidence.

    Return ONLY valid JSON matching this schema:

    {json.dumps(schema, indent=4)}
    """

    original_image_part = types.Part.from_bytes(
        data=base64.b64decode(original_image_base64),
        mime_type="image/jpeg",
    )

    annotated_image_part = types.Part.from_bytes(
        data=base64.b64decode(annotated_base64),
        mime_type="image/jpeg",
    )

    models = [
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-flash-latest",
    "gemini-pro-latest"
     ]

    response = None
    last_error = None

    for model_name in models:
        for attempt in range(5):
            try:
                contents = [
                    prompt,
                    original_image_part
                ]

                if predictions:
                    contents.append(annotated_image_part)
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=1024,
                        response_mime_type="application/json"
                    )
                )

                print(f"Success with model: {model_name}")
                break

            except Exception as ex:
                last_error = ex

                print("="*80)
                print("Gemini Error")
                print(type(ex))
                print(ex)
                print("="*80)

                print(f"{model_name} attempt {attempt+1} failed")
                time.sleep(10)

        if response:
            print(response)
            break

    if not response:
        raise last_error or Exception("All Gemini models failed")

    print(f"Success with model: {model_name}")

    text = (response.text or "").strip()
    print("Gemini Text:")
    print(repr(text))


    if not text:
      raise ValueError("Gemini returned an empty response")

    try:
     print("====================================")
     print("RAW GEMINI RESPONSE")
     print(response)
     print("====================================")

     print("TEXT:")
     print(repr(response.text))
     return json.loads(text)

    except json.JSONDecodeError:

       print("Gemini returned invalid JSON:")
       print(text)

       raise ValueError(
        "Gemini returned invalid JSON"
      )

from datetime import datetime
from zoneinfo import ZoneInfo


def add_ai_watermark(
    image,
    claim,
    evidence_id,
    vehicle,
    gemini_result
):
    """
    Creates a dedicated AI assessment footer below the image.

    Responsive behavior:
    - Wide images: 2-column footer
    - Narrow images: 1-column footer
    """

    # ==================================================
    # IMAGE DIMENSIONS
    # ==================================================

    image_height, image_width = image.shape[:2]

    # ==================================================
    # GEMINI VALUES
    # ==================================================

    severity = str(
        gemini_result.get("severity", "Unknown")
    ).upper()

    cost_min = gemini_result.get(
        "estimatedCostMin", 0
    )

    cost_max = gemini_result.get(
        "estimatedCostMax", 0
    )

    confidence = gemini_result.get(
        "confidence", 0
    )

    # Gemini may return:
    # 0.92
    # or
    # 92

    try:

        confidence = float(confidence)

        if confidence <= 1:
            confidence_percent = round(
                confidence * 100
            )
        else:
            confidence_percent = round(
                confidence
            )

    except (TypeError, ValueError):

        confidence_percent = 0

    # ==================================================
    # VEHICLE
    # ==================================================

    vehicle_name = (
        f"{vehicle.get('make', '')} "
        f"{vehicle.get('model', '')}"
    ).strip()

    if not vehicle_name:

        vehicle_name = "Unknown Vehicle"

    # ==================================================
    # CLAIM
    # ==================================================

    claim_number = claim.get(
        "claimNumber",
        "N/A"
    )

    # ==================================================
    # IST TIME
    # ==================================================

    ist_now = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

    timestamp = ist_now.strftime(
        "%d-%b-%Y %H:%M IST"
    )

    # ==================================================
    # COST
    # ==================================================

    try:

        cost_min = float(cost_min)
        cost_max = float(cost_max)

        estimated_text = (
            f"INR {cost_min:,.0f} - "
            f"INR {cost_max:,.0f}"
        )

    except (TypeError, ValueError):

        estimated_text = "INR 0 - INR 0"

    # ==================================================
    # FONT SIZE BASED ON IMAGE WIDTH
    # ==================================================

    if image_width < 500:

        title_scale = 0.48
        text_scale = 0.30
        thickness = 1
        title_thickness = 2

    elif image_width < 800:

        title_scale = 0.58
        text_scale = 0.36
        thickness = 1
        title_thickness = 2

    else:

        title_scale = 0.70
        text_scale = 0.42
        thickness = 1
        title_thickness = 2

    font = cv2.FONT_HERSHEY_SIMPLEX

    # ==================================================
    # SELECT LAYOUT
    # ==================================================

    # Narrow images MUST use one column.
    # This prevents the overlapping you currently see.

    two_columns = image_width >= 850

    # ==================================================
    # BUILD CONTENT
    # ==================================================

    left_lines = [

        f"Claim      : {claim_number}",

        f"Evidence   : {evidence_id}",

        f"Date       : {timestamp}",

        f"Vehicle    : {vehicle_name}",

        f"Severity   : {severity}"

    ]

    right_lines = [

        f"Estimated  : {estimated_text}",

        f"Confidence : {confidence_percent}%",

        "Gemini 3.6 Flash + Roboflow v1",

        "DO NOT EDIT | DIGITAL EVIDENCE"

    ]

    # ==================================================
    # FOOTER HEIGHT
    # ==================================================

    if two_columns:

        footer_lines = max(
            len(left_lines),
            len(right_lines)
        )

    else:

        footer_lines = (
            len(left_lines) +
            len(right_lines)
        )

    line_height = int(
        image_width * 0.035
    )

    line_height = max(
        18,
        min(line_height, 30)
    )

    footer_height = (
        58 +
        footer_lines * line_height
    )

    # ==================================================
    # CREATE NEW CANVAS
    # ==================================================

    canvas = np.full(
        (
            image_height + footer_height,
            image_width,
            3
        ),
        (25, 25, 25),
        dtype=np.uint8
    )

    # ==================================================
    # ORIGINAL IMAGE
    # ==================================================

    canvas[
        0:image_height,
        0:image_width
    ] = image

    footer_y = image_height

    # ==================================================
    # FOOTER BACKGROUND
    # ==================================================

    cv2.rectangle(
        canvas,

        (0, footer_y),

        (
            image_width,
            image_height + footer_height
        ),

        (25, 25, 25),

        -1
    )

    # ==================================================
    # TOP SEPARATOR
    # ==================================================

    cv2.line(
        canvas,

        (0, footer_y),

        (image_width, footer_y),

        (255, 255, 255),

        2
    )

    # ==================================================
    # HEADER
    # ==================================================

    cv2.putText(
        canvas,

        "AI DAMAGE ASSESSMENT",

        (20, footer_y + 30),

        font,

        title_scale,

        (255, 255, 255),

        title_thickness,

        cv2.LINE_AA
    )

    # ==================================================
    # CONTENT START
    # ==================================================

    start_y = footer_y + 55

    # ==================================================
    # TWO COLUMN LAYOUT
    # ==================================================

    if two_columns:

        left_x = 20

        right_x = int(
            image_width * 0.52
        )

        max_lines = max(
            len(left_lines),
            len(right_lines)
        )

        for i in range(max_lines):

            current_y = (
                start_y +
                i * line_height
            )

            if i < len(left_lines):

                cv2.putText(
                    canvas,

                    left_lines[i],

                    (
                        left_x,
                        current_y
                    ),

                    font,

                    text_scale,

                    (255, 255, 255),

                    thickness,

                    cv2.LINE_AA
                )

            if i < len(right_lines):

                cv2.putText(
                    canvas,

                    right_lines[i],

                    (
                        right_x,
                        current_y
                    ),

                    font,

                    text_scale,

                    (255, 255, 255),

                    thickness,

                    cv2.LINE_AA
                )

    # ==================================================
    # ONE COLUMN LAYOUT
    # ==================================================

    else:

        all_lines = (
            left_lines +
            right_lines
        )

        for i, line in enumerate(all_lines):

            current_y = (
                start_y +
                i * line_height
            )

            cv2.putText(
                canvas,

                line,

                (
                    20,
                    current_y
                ),

                font,

                text_scale,

                (255, 255, 255),

                thickness,

                cv2.LINE_AA
            )

    # ==================================================
    # FOOTER BORDER
    # ==================================================

    cv2.rectangle(
        canvas,

        (8, footer_y + 8),

        (
            image_width - 8,
            image_height + footer_height - 8
        ),

        (180, 180, 180),

        1
    )

    return canvas# --------------------------------------------------
# Detect Endpoint
# --------------------------------------------------

@app.route("/detect", methods=["POST"])
def detect():
    start = time.time()
    temp_path = None

    try:
        data = request.get_json()

        if not data:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "No JSON body received",
                    }
                ),
                400,
            )

        if "image" not in data:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Image not provided",
                    }
                ),
                400,
            )

        vehicle = data.get("vehicle", {})
        claim = data.get("claim", {})
        original_base64 = data["image"]

        # Decode image
        image_bytes = base64.b64decode(original_base64)
        image_np = np.frombuffer(image_bytes, np.uint8)

        image = cv2.imdecode(
            image_np,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Unable to decode image",
                    }
                ),
                400,
            )

        # Determine file extension
        suffix = ".jpg"

        if "fileName" in data:
            _, suffix = os.path.splitext(data["fileName"])

        # Save temporary image
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp:
            temp.write(image_bytes)
            temp.flush()
            temp_path = temp.name

        # Call Roboflow
        with open(temp_path, "rb") as img:
            response = requests.post(
                MODEL_URL,
                params={
                    "api_key": ROBOFLOW_API_KEY,
                },
                files={
                    "file": img,
                },
                timeout=60,
            )

        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            temp_path = None

        # Validate response
        if response.status_code != 200:
            return (
                jsonify(
                    {
                        "success": False,
                        "roboflowStatus": response.status_code,
                        "response": response.text,
                    }
                ),
                response.status_code,
            )

        result = response.json()

        predictions = result.get("predictions", [])

        predictions = [p for p in predictions if p["confidence"] >= 0.35]

        detected_parts = list({p["class"] for p in predictions})

        avg_confidence = 0
        if predictions:
            avg_confidence = round(
                sum(p["confidence"] for p in predictions) / len(predictions), 2
            )

        # Draw detections only if Roboflow found something
        if predictions:
        
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

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

            label = f"{cls} ({conf:.2f})"

            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )

            label_y = max(text_height + 8, y1 - 8)

            cv2.rectangle(
                image,
                (x1, label_y - text_height - 8),
                (x1 + text_width + 8, label_y + baseline),
                color,
                -1,
            )

            cv2.putText(
                image,
                label,
                (x1 + 4, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            


        # Call Gemini for damage estimation
        gemini_result = None
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

            success, buffer = cv2.imencode(".jpg", image, encode_param)
            if not success:
                return (
                    jsonify({"success": False, "message": "Unable to encode image"}),
                    500,
                )
            annotated_base64 = base64.b64encode(buffer).decode("utf-8")
            print("Calling Gemini...")

            gemini_result = estimate_damage_with_gemini(
                annotated_base64, predictions, vehicle, claim, original_base64
            )
            gemini_result["success"] = True
            print("Called Gemini...")

            evidence_id = claim.get(
                        "evidenceNumber",
                      claim.get("Id", "EV-00001")
)
            image = add_ai_watermark(
                    image,
                    claim,
                    evidence_id,
                    vehicle,
                    gemini_result
                )
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            
            success, buffer = cv2.imencode(".jpg", image, encode_param)
            
            if not success:
                        return (
                            jsonify({"success": False, "message": "Unable to encode image"}),
                            500,
                        )
            
            annotated_base64 = base64.b64encode(buffer).decode("utf-8")
            
        except Exception as ex:
            # Don't fail the entire request if Gemini fails; include error info
            gemini_result = {
                 "success": False,
                "error": "Gemini estimation failed",
                "message": str(ex),
                "details": traceback.format_exc(),
            }

        elapsed = round(time.time() - start, 2)

        return jsonify(
            {
                "success": True,
                "predictionCount": len(predictions),
                "predictions": predictions,
                "detectedParts": detected_parts,
                "avgConfidence": avg_confidence,
                "annotatedImage": annotated_base64,
                "gemini": gemini_result,
                "inference_id": result.get("inference_id"),
                "time": result.get("time"),
                "elapsed": elapsed,
            }
        )

    except Exception:
        # Ensure temp cleanup on unexpected errors
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

        return (
            jsonify({"success": False, "message": "Server error", "error": traceback.format_exc()}),
            500,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
