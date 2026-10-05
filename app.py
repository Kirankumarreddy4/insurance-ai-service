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

    # =========================================================
    # STRUCTURED JSON SCHEMA
    # =========================================================

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "summary": {"type": "STRING"},
            "success": {"type": "BOOLEAN"},
            "severity": {"type": "STRING"},
            "recommendation": {"type": "STRING"},
            # Keep these for existing Salesforce mapping
            "partsToReplace": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "partsToRepair": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            # NEW - individual damage items
            "damageItems": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "partName": {"type": "STRING"},
                        "action": {"type": "STRING"},
                        "estimatedCostMin": {"type": "NUMBER"},
                        "estimatedCostMax": {"type": "NUMBER"},
                    },
                    "required": [
                        "partName",
                        "action",
                        "estimatedCostMin",
                        "estimatedCostMax",
                    ],
                },
            },
            "laborHours": {"type": "NUMBER"},
            "estimatedCostMin": {"type": "NUMBER"},
            "estimatedCostMax": {"type": "NUMBER"},
            "damageDetected": {"type": "BOOLEAN"},
            "currency": {"type": "STRING"},
            "confidence": {"type": "NUMBER"},
        },
        "required": [
            "summary",
            "success",
            "severity",
            "recommendation",
            "partsToReplace",
            "partsToRepair",
            "damageItems",
            "laborHours",
            "estimatedCostMin",
            "estimatedCostMax",
            "damageDetected",
            "currency",
            "confidence",
        ],
    }

    # =========================================================
    # PROMPT
    # =========================================================

    prompt = f"""
You are an experienced automobile insurance damage assessor
supporting a motor insurance claims process.

Your task is to analyze the ORIGINAL vehicle image and produce
a structured damage assessment.

Use the Roboflow detections as supporting evidence only.
Do not assume a part is damaged merely because Roboflow detected
an object or class. Visually verify the damage from the ORIGINAL
image.

Vehicle:
{json.dumps(vehicle, indent=2)}

Claim:
{json.dumps(claim, indent=2)}

Roboflow detections:
{json.dumps(predictions, indent=2)}

IMPORTANT ASSESSMENT RULES:

1. Identify only damage that is visibly supported by the image.

2. For every damaged vehicle part, create exactly one item in
   "damageItems".

3. For each damage item provide:
   - partName
   - action
   - estimatedCostMin
   - estimatedCostMax

4. "action" must be exactly one of:
   - "Repair"
   - "Replace"

5. Use realistic repair/replacement cost estimates in INR.

6. Estimate the cost for the individual part itself.
   Do not put the complete vehicle repair cost into each part.

7. Do not double-count the same damaged part.

8. The overall:
   "estimatedCostMin"
   and
   "estimatedCostMax"
   must represent the sum of the corresponding individual
   damage item estimates.

9. If no visible damage is identified:
   - damageItems must be []
   - partsToReplace must be []
   - partsToRepair must be []
   - laborHours must be 0
   - estimatedCostMin must be 0
   - estimatedCostMax must be 0
   - damageDetected must be false
   - severity must be "None"

10. If visible damage exists:
    - damageDetected must be true
    - severity must reflect the overall visible damage
    - include every materially damaged part in damageItems

11. Do not invent hidden or internal damage that cannot be
    reasonably inferred from the image.

12. Distinguish between:
    - Repair: the existing part can reasonably be repaired
    - Replace: the part appears substantially damaged and
      replacement is more appropriate

13. The estimated cost should represent the likely cost for
    repairing or replacing that individual part, including
    reasonable part-related work, but do not add unrelated
    vehicle expenses.

14. Provide laborHours as the estimated labor effort for the
    overall visible damage.

15. Provide confidence as a number between 0 and 1.

16. Compare the visible damage against the claim description.

17. If the claim description does not match the visible damage,
    mention the mismatch clearly in the summary and recommendation.

18. Do not treat a claim-description mismatch by itself as proof
    of fraud.

19. Do not decide insurance coverage, policy eligibility,
    sanction amount, or claim approval/rejection.
    The policy decision will be performed separately by the
    insurance decision engine.

20. Return only the requested structured JSON.

21. Do not return Markdown.

22. Do not add any explanation outside the JSON.

OUTPUT REQUIREMENTS:

The "damageItems" array must contain one object per damaged part.

Example format:

"damageItems": [
    {{
        "partName": "Bonnet / Hood",
        "action": "Replace",
        "estimatedCostMin": 70000,
        "estimatedCostMax": 90000
    }},
    {{
        "partName": "Front Bumper Assembly",
        "action": "Replace",
        "estimatedCostMin": 40000,
        "estimatedCostMax": 50000
    }},
    {{
        "partName": "Front Left Fender",
        "action": "Repair",
        "estimatedCostMin": 10000,
        "estimatedCostMax": 15000
    }}
]

The overall estimatedCostMin and estimatedCostMax must equal
the sum of the corresponding damageItems.

For example, if damageItems contain:

70,000 + 40,000 + 10,000 = 120,000

then estimatedCostMin must be 120000.

Do not include currency symbols or commas inside numeric
values.

Use:
currency = "INR"
"""

    # =========================================================
    # IMAGE PARTS
    # =========================================================

    original_image_part = types.Part.from_bytes(
        data=base64.b64decode(original_image_base64),
        mime_type="image/jpeg",
    )

    annotated_image_part = types.Part.from_bytes(
        data=base64.b64decode(annotated_base64),
        mime_type="image/jpeg",
    )

    # =========================================================
    # GEMINI MODEL FALLBACK 
    # =========================================================

    models = [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite"
    ]

    response = None
    last_error = None

    # =========================================================
    # TRY MODELS
    # =========================================================

    for model_name in models:
        try:
            print(f"Trying Gemini model: {model_name}")

            contents = [prompt, original_image_part]

            if predictions:
                contents.append(annotated_image_part)

            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    max_output_tokens=2048,
                ),
            )

            print(f"Success with model: {model_name}")
            break

        except Exception as ex:
            last_error = ex
            error_text = str(ex)

            print(f"{model_name} failed:")
            print(error_text)

            # =================================================
            # QUOTA / RATE LIMIT
            # =================================================

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "quota" in error_text.lower()
            ):
                print(f"Quota/rate limit on {model_name}. Trying next model.")
                continue

            # =================================================
            # MODEL NOT FOUND
            # =================================================

            if "404" in error_text or "NOT_FOUND" in error_text:
                print(f"{model_name} unavailable. Trying next model.")
                continue

            # =================================================
            # OTHER ERROR
            # =================================================

            print(f"Unexpected Gemini error on {model_name}. Trying next model.")
            continue

    # =========================================================
    # NO MODEL WORKED
    # =========================================================

    if response is None:
        raise last_error or Exception("All Gemini models failed")

    # =========================================================
    # RAW RESPONSE
    # =========================================================

    print("Gemini raw response:")
    print(response.text)

    # =========================================================
    # VALIDATE RESPONSE
    # =========================================================

    if not response.text:
        raise ValueError("Gemini returned an empty response")

    text = response.text.strip()

    # =========================================================
    # PARSE JSON
    # =========================================================

    try:
        result = json.loads(text)
    except json.JSONDecodeError as json_error:
        print("Gemini returned invalid JSON:")
        print(text)
        print("JSON error:", json_error)
        raise ValueError("Gemini returned invalid JSON")

    print("Gemini JSON parsed successfully")
    return result

# --------------------------------------------------
# Gemini Claim Decision Analysis
# --------------------------------------------------

def analyze_claim_decision_with_gemini(
    claim,
    damage_lines,
    policy_terms
):
    """
    Second-stage insurance claim decision analysis.

    IMPORTANT:
    - No images are sent here.
    - Damage assessment was already completed by /detect.
    - This stage analyzes damage lines + policy terms.
    - Returns compact structured JSON.
    """

    # =========================================================
    # STRUCTURED JSON SCHEMA
    # =========================================================

    response_schema = {
        "type": "OBJECT",
        "properties": {

            "claimSummary": {
                "type": "STRING"
            },

            "coverageOverview": {
                "type": "STRING"
            },

            "financialInsight": {
                "type": "STRING"
            },

            "missingInformation": {
                "type": "STRING"
            },

            "policyConflicts": {
                "type": "STRING"
            },

            "humanReviewRequired": {
                "type": "BOOLEAN"
            },

            "overallConfidence": {
                "type": "NUMBER"
            },

            "damageLineInsights": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {

                        "partName": {
                            "type": "STRING"
                        },

                        "action": {
                            "type": "STRING"
                        },

                        "coverageStatus": {
                            "type": "STRING"
                        },

                        "applicableClauseNumber": {
                            "type": "STRING"
                        },

                        "applicableClauseTitle": {
                            "type": "STRING"
                        },

                        "decisionImpact": {
                            "type": "STRING"
                        },

                        "coveredAmount": {
                            "type": "NUMBER"
                        },

                        "nonCoveredAmount": {
                            "type": "NUMBER"
                        },

                        "deductibleAmount": {
                            "type": "NUMBER"
                        },

                        "policyLimitAmount": {
                            "type": "NUMBER"
                        },

                        "reason": {
                            "type": "STRING"
                        },

                        "sourcePageNumber": {
                            "type": "NUMBER"
                        },

                        "confidence": {
                            "type": "NUMBER"
                        }
                    },

                    "required": [
                        "partName",
                        "action",
                        "coverageStatus",
                        "applicableClauseNumber",
                        "applicableClauseTitle",
                        "decisionImpact",
                        "coveredAmount",
                        "nonCoveredAmount",
                        "deductibleAmount",
                        "policyLimitAmount",
                        "reason",
                        "sourcePageNumber",
                        "confidence"
                    ]
                }
            }
        },

        "required": [
            "claimSummary",
            "coverageOverview",
            "financialInsight",
            "missingInformation",
            "policyConflicts",
            "humanReviewRequired",
            "overallConfidence",
            "damageLineInsights"
        ]
    }


    # =========================================================
    # COMPACT DAMAGE DATA
    # =========================================================

    compact_damage_lines = []

    for line in damage_lines or []:

        compact_damage_lines.append({

            "damageLineId":
                line.get("damageLineId"),

            "partName":
                line.get("partName"),

            "action":
                line.get("action"),

            "estimatedCostMin":
                line.get("aiEstimatedCostMin"),

            "estimatedCostMax":
                line.get("aiEstimatedCostMax"),

            "severity":
                line.get("aiSeverity"),

            "confidence":
                line.get("aiConfidence"),

            "summary":
                line.get("aiSummary"),

            "recommendation":
                line.get("aiRecommendation")
        })


    # =========================================================
    # COMPACT POLICY DATA
    # =========================================================

    compact_policy_terms = []

    for term in policy_terms or []:

        compact_policy_terms.append({

            "clauseNumber":
                term.get("clauseNumber"),

            "clauseTitle":
                term.get("clauseTitle"),

            "clauseType":
                term.get("clauseType"),

            "description":
                term.get("description"),

            "applicablePart":
                term.get("applicablePart"),

            "generalTerm":
                term.get("generalTerm"),

            "limitAmount":
                term.get("limitAmount"),

            "deductibleAmount":
                term.get("deductibleAmount"),

            "deductiblePercentage":
                term.get("deductiblePercentage"),

            "maximumPayableAmount":
                term.get("maximumPayableAmount"),

            "decisionImpact":
                term.get("decisionImpact"),

            "sourcePageNumber":
                term.get("sourcePageNumber")
        })


    # =========================================================
    # COMPACT JSON FOR PROMPT
    # =========================================================

    claim_json = json.dumps(
        claim or {},
        separators=(",", ":"),
        ensure_ascii=False
    )

    damage_json = json.dumps(
        compact_damage_lines,
        separators=(",", ":"),
        ensure_ascii=False
    )

    policy_json = json.dumps(
        compact_policy_terms,
        separators=(",", ":"),
        ensure_ascii=False
    )


    # =========================================================
    # PROMPT
    # =========================================================

    prompt = f"""
You are an automobile insurance claim decision analysis engine.

Analyze the supplied claim, AI damage assessment and policy terms.

Your role is to provide a policy-analysis recommendation for a
human Service Representative.

Do NOT make the final claim approval decision.

CLAIM:
{claim_json}

DAMAGE LINES:
{damage_json}

POLICY TERMS:
{policy_json}

RULES:

1. Use ONLY the supplied policy terms.

2. Do NOT invent clause numbers, clause titles, exclusions,
   deductibles, limits or policy conditions.

3. For each damage line return exactly ONE damageLineInsights item.

4. Match each damaged part against the supplied policy terms.

5. General policy terms may apply to a damaged part.

6. The absence of a matching clause is NOT automatically proof
   that the damage is excluded.

7. If a policy term clearly supports coverage, use:
   Covered

8. If the supplied terms indicate only partial coverage or a
   restricted limit, use:
   Partially Covered

9. If the supplied policy terms explicitly exclude the damage,
   use:
   Not Covered

10. If the policy information is insufficient or contradictory,
    use:
    Requires Review

11. coverageStatus must be exactly one of:
    Covered
    Partially Covered
    Not Covered
    Requires Review

12. action must come from the supplied damage line.

13. applicableClauseNumber must exactly match a supplied
    policy clause number.
    If no applicable clause exists, return "".

14. applicableClauseTitle must exactly match the supplied
    policy clause title.
    If no applicable clause exists, return "".

15. Do not invent numeric values.

16. coveredAmount means the amount supported by policy terms
    for the damaged item, NOT the final claim payout.

17. nonCoveredAmount means the amount identified as not covered
    based on the supplied policy terms.

18. deductibleAmount must be the supplied policy deductible
    applicable to the damage.
    If not available, return 0.

19. policyLimitAmount must use only a supplied policy limit.
    If not available, return 0.

20. Final payable claim amount must NOT be calculated here.

21. The Service Representative will later enter the Assessed
    Repair Amount and the final payable amount will be determined
    separately.

22. If claim description conflicts with observed damage,
    clearly mention the mismatch in claimSummary or
    policyConflicts.

23. A claim-description mismatch alone is NOT proof of fraud.

24. Keep claimSummary concise.

25. Keep coverageOverview concise.

26. Keep financialInsight concise.

27. Keep missingInformation concise.

28. Keep policyConflicts concise.

29. Keep each damage-line reason concise.

30. Do not return Markdown.

31. Return ONLY valid JSON matching the provided schema.

32. Use empty strings for unavailable text values.

33. Use 0 for unavailable numeric values.

TEXT LENGTH LIMITS:

claimSummary: maximum approximately 350 characters.

coverageOverview: maximum approximately 350 characters.

financialInsight: maximum approximately 350 characters.

missingInformation: maximum approximately 300 characters.

policyConflicts: maximum approximately 400 characters.

reason for each damage line: maximum approximately 250 characters.

IMPORTANT:

Do not write long explanations.

Return all required fields.
"""


    # =========================================================
    # MODEL ORDER
    # =========================================================
    #
    # Do NOT use gemini-3.8-flash first here.
    # Your Render logs already showed 503/high demand.
    #
    # 3.5 Flash = primary
    # 3.5 Flash-Lite = fallback
    #

    models = [
        (
            "gemini-3.5-flash",
            "low"
        ),
        (
            "gemini-3.5-flash-lite",
            "minimal"
        )
    ]


    last_error = None
    last_json_error = None


    # =========================================================
    # TRY MODELS
    # =========================================================

    for model_name, thinking_level in models:

        try:

            print(
                f"Trying Gemini decision model: {model_name}"
            )

            response = client.models.generate_content(

                model=model_name,

                contents=prompt,

                config=types.GenerateContentConfig(

                    response_mime_type="application/json",

                    response_schema=response_schema,

                    max_output_tokens=4096,

                    thinking_config=types.ThinkingConfig(
                        thinking_level=thinking_level
                    )
                )
            )


            # =================================================
            # FINISH REASON
            # =================================================

            finish_reason = "UNKNOWN"

            try:

                if response.candidates:

                    finish_reason = str(
                        response.candidates[0].finish_reason
                    )

            except Exception:
                pass


            print(
                f"Gemini decision model finished: "
                f"{model_name} | "
                f"finish_reason={finish_reason}"
            )


            # =================================================
            # GET TEXT
            # =================================================

            text = (
                response.text or ""
            ).strip()


            print(
                f"Gemini decision response length: "
                f"{len(text)}"
            )


            if not text:

                raise ValueError(
                    "Gemini returned an empty decision response"
                )


            # =================================================
            # REMOVE OPTIONAL MARKDOWN FENCE
            # =================================================

            if text.startswith("```"):

                text = re.sub(
                    r"^```(?:json)?\s*",
                    "",
                    text,
                    flags=re.IGNORECASE
                )

                text = re.sub(
                    r"\s*```$",
                    "",
                    text
                ).strip()


            # =================================================
            # PARSE JSON
            # =================================================

            try:

                result = json.loads(
                    text
                )

            except json.JSONDecodeError as json_error:

                last_json_error = json_error

                print(
                    "Gemini returned invalid decision JSON."
                )

                print(
                    f"JSON error: {json_error}"
                )

                print(
                    f"Finish reason: {finish_reason}"
                )

                print(
                    f"Response length: {len(text)}"
                )

                # IMPORTANT:
                # Do NOT stop here.
                # Try the fallback model.

                continue


            # =================================================
            # BASIC RESPONSE VALIDATION
            # =================================================

            required_keys = [

                "claimSummary",
                "coverageOverview",
                "financialInsight",
                "missingInformation",
                "policyConflicts",
                "humanReviewRequired",
                "overallConfidence",
                "damageLineInsights"
            ]


            missing_keys = [

                key

                for key in required_keys

                if key not in result
            ]


            if missing_keys:

                print(
                    "Gemini decision response missing keys: " +
                    str(missing_keys)
                )

                last_error = ValueError(
                    "Gemini decision response missing required fields: " +
                    ", ".join(missing_keys)
                )

                continue


            if not isinstance(
                result.get("damageLineInsights"),
                list
            ):

                last_error = ValueError(
                    "damageLineInsights must be an array"
                )

                continue


            # =================================================
            # NORMALIZE DAMAGE LINE VALUES
            # =================================================

            for insight in result["damageLineInsights"]:

                if not isinstance(
                    insight,
                    dict
                ):
                    continue


                if not insight.get(
                    "applicableClauseNumber"
                ):

                    insight[
                        "applicableClauseNumber"
                    ] = ""


                if not insight.get(
                    "applicableClauseTitle"
                ):

                    insight[
                        "applicableClauseTitle"
                    ] = ""


                if not insight.get(
                    "decisionImpact"
                ):

                    insight[
                        "decisionImpact"
                    ] = ""


                if not insight.get(
                    "coveredAmount"
                ):

                    insight[
                        "coveredAmount"
                    ] = 0


                if not insight.get(
                    "nonCoveredAmount"
                ):

                    insight[
                        "nonCoveredAmount"
                    ] = 0


                if not insight.get(
                    "deductibleAmount"
                ):

                    insight[
                        "deductibleAmount"
                    ] = 0


                if not insight.get(
                    "policyLimitAmount"
                ):

                    insight[
                        "policyLimitAmount"
                    ] = 0


                if not insight.get(
                    "reason"
                ):

                    insight[
                        "reason"
                    ] = ""


                if not insight.get(
                    "sourcePageNumber"
                ):

                    insight[
                        "sourcePageNumber"
                    ] = 0


                if not insight.get(
                    "confidence"
                ):

                    insight[
                        "confidence"
                    ] = 0


            print(
                "Claim decision analysis succeeded with: " +
                model_name
            )


            print(
                "Gemini decision JSON parsed successfully."
            )


            return result


        except Exception as ex:

            last_error = ex

            error_text = str(ex)


            print(
                f"Decision model {model_name} failed:"
            )

            print(
                error_text
            )


            # =================================================
            # FALLBACK CONDITIONS
            # =================================================

            if (
                "429" in error_text
                or
                "RESOURCE_EXHAUSTED"
                in error_text
                or
                "503" in error_text
                or
                "UNAVAILABLE" in error_text
                or
                "404" in error_text
                or
                "NOT_FOUND" in error_text
            ):

                print(
                    f"Trying next decision model "
                    f"after {model_name}."
                )

                continue


            # Continue to fallback for unexpected errors too.
            print(
                f"Unexpected decision error on "
                f"{model_name}. Trying next model."
            )

            continue


    # =========================================================
    # NOTHING WORKED
    # =========================================================

    if last_json_error is not None:

        raise ValueError(
            "Gemini returned invalid decision JSON after trying "
            "all decision models. Last JSON error: " +
            str(last_json_error)
        )


    if last_error is not None:

        raise last_error


    raise ValueError(
        "All Gemini decision models failed."
    )
from datetime import datetime
from zoneinfo import ZoneInfo


def add_ai_watermark(image, claim, evidence_id, vehicle, gemini_result):
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

    severity = str(gemini_result.get("severity", "Unknown")).upper()
    cost_min = gemini_result.get("estimatedCostMin", 0)
    cost_max = gemini_result.get("estimatedCostMax", 0)
    confidence = gemini_result.get("confidence", 0)

    # Gemini may return:
    # 0.92
    # or
    # 92
    try:
        confidence = float(confidence)

        if confidence <= 1:
            confidence_percent = round(confidence * 100)
        else:
            confidence_percent = round(confidence)
    except (TypeError, ValueError):
        confidence_percent = 0

    # ==================================================
    # VEHICLE
    # ==================================================

    vehicle_name = (f"{vehicle.get('make', '')} {vehicle.get('model', '')}").strip()

    if not vehicle_name:
        vehicle_name = "Unknown Vehicle"

    # ==================================================
    # CLAIM
    # ==================================================

    claim_number = claim.get("claimNumber", "N/A")

    # ==================================================
    # IST TIME
    # ==================================================

    ist_now = datetime.now(ZoneInfo("Asia/Kolkata"))
    timestamp = ist_now.strftime("%d-%b-%Y %H:%M IST")

    # ==================================================
    # COST
    # ==================================================

    try:
        cost_min = float(cost_min)
        cost_max = float(cost_max)
        estimated_text = f"INR {cost_min:,.0f} - INR {cost_max:,.0f}"
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
        f"Severity   : {severity}",
    ]

    right_lines = [
        f"Estimated  : {estimated_text}",
        f"Confidence : {confidence_percent}%",
        "Gemini 3.6 Flash + Roboflow v1",
        "DO NOT EDIT | DIGITAL EVIDENCE",
    ]

    # ==================================================
    # FOOTER HEIGHT
    # ==================================================

    if two_columns:
        footer_lines = max(len(left_lines), len(right_lines))
    else:
        footer_lines = len(left_lines) + len(right_lines)

    line_height = int(image_width * 0.035)
    line_height = max(18, min(line_height, 30))
    footer_height = 58 + footer_lines * line_height

    # ==================================================
    # CREATE NEW CANVAS
    # ==================================================

    canvas = np.full(
        (image_height + footer_height, image_width, 3),
        (25, 25, 25),
        dtype=np.uint8,
    )

    # ==================================================
    # ORIGINAL IMAGE
    # ==================================================

    canvas[0:image_height, 0:image_width] = image
    footer_y = image_height

    # ==================================================
    # FOOTER BACKGROUND
    # ==================================================

    cv2.rectangle(
        canvas,
        (0, footer_y),
        (image_width, image_height + footer_height),
        (25, 25, 25),
        -1,
    )

    # ==================================================
    # TOP SEPARATOR
    # ==================================================

    cv2.line(
        canvas,
        (0, footer_y),
        (image_width, footer_y),
        (255, 255, 255),
        2,
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
        cv2.LINE_AA,
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
        right_x = int(image_width * 0.52)
        max_lines = max(len(left_lines), len(right_lines))

        for i in range(max_lines):
            current_y = start_y + i * line_height

            if i < len(left_lines):
                cv2.putText(
                    canvas,
                    left_lines[i],
                    (left_x, current_y),
                    font,
                    text_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

            if i < len(right_lines):
                cv2.putText(
                    canvas,
                    right_lines[i],
                    (right_x, current_y),
                    font,
                    text_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

    # ==================================================
    # ONE COLUMN LAYOUT
    # ==================================================

    else:
        all_lines = left_lines + right_lines

        for i, line in enumerate(all_lines):
            current_y = start_y + i * line_height
            cv2.putText(
                canvas,
                line,
                (20, current_y),
                font,
                text_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    # ==================================================
    # FOOTER BORDER
    # ==================================================

    cv2.rectangle(
        canvas,
        (8, footer_y + 8),
        (image_width - 8, image_height + footer_height - 8),
        (180, 180, 180),
        1,
    )

    return canvas


# --------------------------------------------------
# Detect Endpoint
# --------------------------------------------------
# --------------------------------------------------
# Claim Decision Endpoint
# --------------------------------------------------

@app.route("/claim-decision", methods=["POST"])
def claim_decision():

    start = time.time()

    try:

        # =================================================
        # READ REQUEST
        # =================================================

        data = request.get_json()


        if not data:

            return jsonify({
                "success": False,
                "message": "No JSON body received"
            }), 400


        # =================================================
        # REQUIRED INPUTS
        # =================================================

        claim = data.get(
            "claim",
            {}
        )

        damage_lines = data.get(
            "damageLines",
            []
        )

        policy_terms = data.get(
            "policyTerms",
            []
        )


        if not claim:

            return jsonify({
                "success": False,
                "message": "Claim information is required"
            }), 400


        if not damage_lines:

            return jsonify({
                "success": False,
                "message": "damageLines are required"
            }), 400


        if not policy_terms:

            return jsonify({
                "success": False,
                "message": "policyTerms are required"
            }), 400


        # =================================================
        # LOG INPUT
        # =================================================

        print(
            "Claim Decision Request"
        )

        print(
            "Claim:",
            json.dumps(
                claim,
                indent=2
            )
        )

        print(
            "Damage Lines Count:",
            len(damage_lines)
        )

        print(
            "Policy Terms Count:",
            len(policy_terms)
        )


        # =================================================
        # GEMINI DECISION ANALYSIS
        # =================================================

        decision_result = (
            analyze_claim_decision_with_gemini(
                claim,
                damage_lines,
                policy_terms
            )
        )


        # =================================================
        # SUCCESS
        # =================================================

        elapsed = round(
            time.time() - start,
            2
        )


        return jsonify({

            "success": True,

            "claimDecision": decision_result,

            "damageLineCount":
                len(damage_lines),

            "policyTermCount":
                len(policy_terms),

            "elapsed":
                elapsed

        }), 200


    # =====================================================
    # ERROR
    # =====================================================

    except Exception as ex:

        print(
            "Claim decision analysis failed:"
        )

        print(
            str(ex)
        )

        print(
            traceback.format_exc()
        )


        return jsonify({

            "success": False,

            "message":
                "Claim decision analysis failed",

            "error":
                str(ex),

            "details":
                traceback.format_exc()

        }), 500
@app.route("/detect", methods=["POST"])
def detect():
    start = time.time()
    temp_path = None
    result = {}
    annotated_base64 = ""

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
                annotated_base64,
                predictions,
                vehicle,
                claim,
                original_base64,
            )

            # =========================================================
            # VALIDATE DAMAGE ITEMS
            # =========================================================

            damageItems = gemini_result.get("damageItems", [])

            for item in damageItems:
                if item.get("action") not in ["Repair", "Replace"]:
                    raise ValueError(
                        f"Invalid damage action: {item.get('action')}"
                    )

                if not item.get("partName"):
                    raise ValueError("Damage item missing partName")

                cost_min = float(item.get("estimatedCostMin", 0))
                cost_max = float(item.get("estimatedCostMax", 0))

                if cost_min < 0:
                    raise ValueError("Damage item estimatedCostMin cannot be negative")

                if cost_max < 0:
                    raise ValueError("Damage item estimatedCostMax cannot be negative")

                if cost_min > cost_max:
                    raise ValueError(
                        f"Invalid cost range for {item.get('partName')}"
                    )

            # =========================================================
            # VALIDATE TOTAL COST AGAINST DAMAGE ITEMS
            # =========================================================

            calculated_min = sum(
                float(item.get("estimatedCostMin", 0))
                for item in damageItems
            )

            calculated_max = sum(
                float(item.get("estimatedCostMax", 0))
                for item in damageItems
            )

            gemini_min = float(gemini_result.get("estimatedCostMin", 0))
            gemini_max = float(gemini_result.get("estimatedCostMax", 0))

            if abs(calculated_min - gemini_min) > 1:
                raise ValueError(
                    f"estimatedCostMin mismatch. Damage items total = {calculated_min}, Gemini total = {gemini_min}"
                )

            if abs(calculated_max - gemini_max) > 1:
                raise ValueError(
                    f"estimatedCostMax mismatch. Damage items total = {calculated_max}, Gemini total = {gemini_max}"
                )

            # =========================================================
            # GEMINI SUCCESS
            # =========================================================

            gemini_result["success"] = True
            print("Called Gemini...")

            evidence_id = claim.get(
                "evidenceNumber",
                claim.get("Id", "EV-00001"),
            )
            image = add_ai_watermark(
                image,
                claim,
                evidence_id,
                vehicle,
                gemini_result,
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
