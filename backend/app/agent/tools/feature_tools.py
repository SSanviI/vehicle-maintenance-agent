"""
Feature Tools — Deterministic Python validation and state updates.
Validates LLM extractions against dataset schema, enforces canonical values,
resolves current question context, tolerates spelling errors, parses dates,
and prevents unconfirmed values from entering state.
"""

from __future__ import annotations

import re
import difflib
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from backend.app.ml.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_VALUES,
    FEATURE_DISPLAY_NAMES,
)
from backend.app.agent.state.state import AgentState
from backend.app.ml.feature_schema import MODEL_FEATURE_SCHEMA as SELECTED_FEATURES, DATE_FEATURES, is_valid_feature
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

_UNKNOWN_PATTERNS = (
    "i don't know", "i dont know", "not sure", "no idea", "can't remember",
    "cannot remember", "dont remember", "don't remember", "no clue", "unclear",
    "not known", "unknown", "unsure"
)

_SUMMARY_YES_PATTERNS = (
    "yes", "yeah", "yep", "sure", "please", "yes please", "yes explain", "explain please",
    "show me", "give me the summary", "okay", "ok", "i'd like that", "show summary", "summary",
    "yes summary", "yes tell me", "tell me", "explain", "details", "yes i would like", "yes i would like to see it",
    "i would like to", "i'd like to", "yes please explain"
)

_SUMMARY_NO_PATTERNS = (
    "no", "no thanks", "not now", "that's enough", "i'm good", "nope", "no thank you",
    "nah", "dont need it", "don't need it", "stop"
)


def classify_post_assessment_intent(user_message: str, previous_mode: str = "AWAITING_SUMMARY_CONFIRMATION") -> str:
    """
    Classify user intent during post-assessment conversational turns.
    Returns:
      "SUMMARY_REQUEST"
      "SUMMARY_DECLINE"
      "FACTOR_EXPLANATION"
      "RISK_EXPLANATION"
      "ASSESSMENT_DETAILS"
      "RECOMMENDATION"
      "CAPABILITY_QUESTION"
      "GENERAL_FOLLOW_UP"
    """
    clean = re.sub(r"[^\w\s]", " ", (user_message or "").lower()).strip()
    clean_compact = " ".join(clean.split())
    if not clean_compact:
        return "GENERAL_FOLLOW_UP"

    # An explicit request to start over is the only post-assessment request
    # allowed to clear the canonical assessment facts.
    if any(p in clean_compact for p in (
        "new assessment", "start a new assessment", "start new assessment",
        "start over", "restart assessment", "another assessment", "assess another"
    )):
        return "NEW_ASSESSMENT"

    # 1. Capability Boundaries (RUL, Causal claims)
    if any(p in clean_compact for p in (
        "remaining useful life", "useful life", "years left", "how many years", "remaining life",
        "caused the failure", "cause of failure", "why did it fail", "failure cause", "root cause",
        "what caused"
    )):
        return "CAPABILITY_QUESTION"

    # 2. Factor Explanation (What is the main factor / what affected the score / what drove the score)
    if any(p in clean_compact for p in (
        "main factor", "top factor", "most important factor", "strongest factor",
        "which factor", "what factor", "what affected", "what influenced", "which feature",
        "what contributed", "factor affected", "factor mattered", "factors behind", "which one",
        "what drove", "affected the score", "affected score", "affected my score",
        "what changed the score", "what makes the score", "what caused the score",
        "primary factor", "leading factor", "key factor", "why is this factor important",
        "why did this affect", "why this feature", "explain this factor"
    )):
        return "FACTOR_EXPLANATION"

    # 3. Risk Explanation (Why is the risk low / high / what is the reason for this output)
    if any(p in clean_compact for p in (
        "why is the risk", "why is risk", "why low", "why high", "why did it predict",
        "reason for risk", "reason for this output", "reason for this result",
        "reason for the output", "reason for the result", "reason for my result",
        "why this output", "why this result", "why the score", "why is it low",
        "why is it high", "why is my risk", "what is the reason", "what's the reason",
        "why is the score", "why the prediction", "why maintenance required",
        "why did i get this result", "why did i get this", "why this result",
        "why is my result", "why did i receive this", "what caused this result",
        "why is my score", "why is it showing", "why is the output"
    )) or (clean_compact in ("why", "why?", "how so", "how come", "why this?")):
        return "RISK_EXPLANATION"


    # Short replies are meaningful after a result has been shown.  They should
    # continue the result discussion rather than fall through to the completed
    # assessment closing prompt.
    if previous_mode in ("SUMMARY", "EXPLANATION", "FOLLOW_UP") and clean_compact in (
        "explain", "explain it", "tell me", "how", "how?", "what affected it"
    ):
        return "RISK_EXPLANATION"

    # 4. Assessment Details / Features Used
    if any(p in clean_compact for p in (
        "what information did you use", "what details did you use", "what data did you use",
        "what features were used", "what did you base"
    )):
        return "ASSESSMENT_DETAILS"

    # 5. Recommendation / Next Steps
    if any(p in clean_compact for p in (
        "what should i do", "what should i check", "what to check", "check first",
        "reduce risk", "lower risk", "next steps", "recommendation", "what now", "what next"
    )):
        return "RECOMMENDATION"

    # 6. Decline
    if clean_compact in _SUMMARY_NO_PATTERNS or any(clean_compact == p for p in _SUMMARY_NO_PATTERNS) or any(clean_compact.startswith(p) for p in ("no ", "not now", "nope", "nah")):
        return "SUMMARY_DECLINE"

    # 7. Summary Confirmation / Request
    if previous_mode == "AWAITING_SUMMARY_CONFIRMATION":
        if clean_compact in _SUMMARY_YES_PATTERNS or any(clean_compact == p for p in _SUMMARY_YES_PATTERNS) or any(clean_compact.startswith(p) for p in ("yes", "yeah", "yep", "sure", "please", "show", "give", "explain", "i'd like")):
            return "SUMMARY_REQUEST"
    else:
        if any(p in clean_compact for p in ("summary", "recap", "give me summary", "show summary", "see summary")):
            return "SUMMARY_REQUEST"

    return "GENERAL_FOLLOW_UP"


def classify_summary_intent(user_message: str) -> Optional[str]:
    """Backward-compatible helper returning 'yes' or 'no'."""
    intent = classify_post_assessment_intent(user_message, "AWAITING_SUMMARY_CONFIRMATION")
    if intent == "SUMMARY_REQUEST":
        return "yes"
    if intent == "SUMMARY_DECLINE":
        return "no"
    return None

_NUMBER_WORDS = {
    "zero": 0, "none": 0, "no": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20, "thirty": 30,
}

_DATE_FORMATS = (
    "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%d %B %Y", "%d %b %Y", "%d %B, %Y", "%d %b, %Y",
    "%B %d %Y", "%b %d %Y", "%B %d, %Y", "%b %d, %Y",
    "%Y/%m/%d", "%m/%d/%Y",
)

# Canonical feature aliases and typo mappings
_FEATURE_NAME_ALIASES: Dict[str, str] = {
    # Brake_Condition
    "brake": "Brake_Condition", "brakes": "Brake_Condition", "brke": "Brake_Condition",
    "brkes": "Brake_Condition", "breaks": "Brake_Condition", "break": "Brake_Condition",
    "brake condition": "Brake_Condition", "brakes condition": "Brake_Condition",
    # Battery_Status
    "battery": "Battery_Status", "batteru": "Battery_Status", "batery": "Battery_Status",
    "battery status": "Battery_Status", "battery condition": "Battery_Status",
    # Maintenance_History
    "maintenance": "Maintenance_History", "maintainance": "Maintenance_History",
    "maintenence": "Maintenance_History", "maintanence": "Maintenance_History",
    "maintenance history": "Maintenance_History", "maintainance history": "Maintenance_History",
    # Service_History
    "service": "Service_History", "services": "Service_History", "service history": "Service_History",
    "service visits": "Service_History", "serviced": "Service_History", "servicing": "Service_History",
    # Reported_Issues
    "reported issues": "Reported_Issues", "issues": "Reported_Issues", "problems": "Reported_Issues",
    "reported problems": "Reported_Issues", "faults": "Reported_Issues", "issues count": "Reported_Issues",
    # Vehicle_Age
    "vehicle age": "Vehicle_Age", "age": "Vehicle_Age", "car age": "Vehicle_Age", "bike age": "Vehicle_Age",
    # Tire_Condition
    "tire": "Tire_Condition", "tires": "Tire_Condition", "tyre": "Tire_Condition", "tyres": "Tire_Condition",
    "tire condition": "Tire_Condition", "tyre condition": "Tire_Condition",
    # Accident_History
    "accident": "Accident_History", "accidents": "Accident_History", "accident history": "Accident_History",
    "crashes": "Accident_History",
    # Last_Service_Date
    "last service": "Last_Service_Date", "last service date": "Last_Service_Date",
    "last serviced": "Last_Service_Date", "service date": "Last_Service_Date",
    # Warranty_Expiry_Date
    "warranty": "Warranty_Expiry_Date", "warranty expiry": "Warranty_Expiry_Date",
    "warranty expiry date": "Warranty_Expiry_Date", "warranty date": "Warranty_Expiry_Date",
    # Numerical features
    "mileage": "Mileage", "odometer": "Odometer_Reading", "odometer reading": "Odometer_Reading",
    "engine size": "Engine_Size", "engine": "Engine_Size", "fuel efficiency": "Fuel_Efficiency",
    "insurance premium": "Insurance_Premium", "insurance": "Insurance_Premium",
    # Categorical features
    "fuel type": "Fuel_Type", "fuel": "Fuel_Type", "transmission": "Transmission_Type",
    "transmission type": "Transmission_Type", "owner type": "Owner_Type", "owner": "Owner_Type",
    "vehicle model": "Vehicle_Model", "vehicle type": "Vehicle_Model", "model": "Vehicle_Model",
}

# Categorical synonyms and typo dictionary
_CATEGORICAL_SYNONYMS: Dict[str, Dict[str, str]] = {
    "Maintenance_History": {
        "good": "Good", "gud": "Good", "great": "Good", "well maintained": "Good", "excellent": "Good", "regular": "Good", "fine": "Good",
        "average": "Average", "avrage": "Average", "avg": "Average", "moderate": "Average", "ok": "Average", "okay": "Average",
        "fair": "Average", "normal": "Average", "decent": "Average", "medium": "Average", "mediocre": "Average",
        "poor": "Poor", "bad": "Poor", "neglected": "Poor", "terrible": "Poor", "rough": "Poor", "irregular": "Poor", "none": "Poor",
    },
    "Brake_Condition": {
        "new": "New", "brand new": "New", "replaced": "New", "recently replaced": "New",
        "good": "Good", "gud": "Good", "fine": "Good", "ok": "Good", "okay": "Good", "decent": "Good", "working": "Good", "great": "Good",
        "worn out": "Worn Out", "worn": "Worn Out", "wornout": "Worn Out", "bad": "Worn Out", "weak": "Worn Out", "soft": "Worn Out",
        "terrible": "Worn Out", "poor": "Worn Out",
    },

    "Battery_Status": {
        "new": "New", "brand new": "New", "replaced": "New", "recently replaced": "New",
        "good": "Good", "gud": "Good", "fine": "Good", "ok": "Good", "okay": "Good", "healthy": "Good", "strong": "Good", "great": "Good",
        "weak": "Weak", "wek": "Weak", "bad": "Weak", "low": "Weak", "poor": "Weak", "dead": "Weak", "failing": "Weak",
    },
    "Tire_Condition": {
        "new": "New", "brand new": "New", "replaced": "New", "recently replaced": "New",
        "good": "Good", "gud": "Good", "fine": "Good", "ok": "Good", "okay": "Good", "decent": "Good", "great": "Good",
        "worn out": "Worn Out", "worn": "Worn Out", "wornout": "Worn Out", "bad": "Worn Out", "bald": "Worn Out", "smooth": "Worn Out", "poor": "Worn Out",
    },
    "Fuel_Type": {
        "petrol": "Petrol", "gas": "Petrol", "gasoline": "Petrol",
        "diesel": "Diesel",
        "electric": "Electric", "ev": "Electric",
    },
    "Transmission_Type": {
        "automatic": "Automatic", "auto": "Automatic", "at": "Automatic",
        "manual": "Manual", "stick": "Manual", "mt": "Manual", "standard": "Manual",
    },
    "Owner_Type": {
        "first": "First", "1st": "First", "1": "First", "single owner": "First", "one": "First",
        "second": "Second", "2nd": "Second", "2": "Second", "two": "Second",
        "third": "Third", "3rd": "Third", "3": "Third", "three": "Third", "fourth": "Third", "4th": "Third",
    },
    "Vehicle_Model": {
        "car": "Car", "sedan": "Car", "hatchback": "Car",
        "suv": "SUV", "crossover": "SUV",
        "truck": "Truck", "pickup": "Truck",
        "van": "Van", "minivan": "Van",
        "motorcycle": "Motorcycle", "bike": "Motorcycle", "scooter": "Motorcycle",
        "bus": "Bus",
    },
}


def normalise_date_value(raw_value: Any) -> Optional[str]:
    """Return a real calendar date as the model's ISO-8601 representation (YYYY-MM-DD)."""
    if not raw_value:
        return None
    value = str(raw_value).strip()
    # Strip ordinal suffixes (15th -> 15, 1st -> 1, 2nd -> 2, 3rd -> 3)
    value = re.sub(r"(\d{1,2})(?:st|nd|rd|th)\b", r"\1", value, flags=re.IGNORECASE)
    # Remove leading date phrases like "last serviced on", "serviced on", "date:", etc.
    value = re.sub(r"^(?:last\s+serviced\s+(?:on\s+)?|serviced\s+(?:on\s+)?|last\s+service\s+(?:date\s+)?(?:is|=|:)?\s*|warranty\s+(?:expiry\s+)?(?:date\s+)?(?:is|=|:)?\s*|date\s*(?:is|=|:)?\s*)", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"\s+", " ", value)

    for date_format in _DATE_FORMATS:
        try:
            return datetime.strptime(value, date_format).date().isoformat()
        except ValueError:
            continue

    # Try matching regex date within string
    patterns = [
        r"(\d{4}-\d{1,2}-\d{1,2})",
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{4})",
        r"(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})",
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, value, re.IGNORECASE)
        if m:
            sub_val = m.group(1).replace(",", "")
            for date_format in _DATE_FORMATS:
                try:
                    return datetime.strptime(sub_val, date_format).date().isoformat()
                except ValueError:
                    continue
    return None


def validate_feature_name(feature_name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate whether a feature belongs strictly to the dataset schema using central validation.
    Tolerates spelling mistakes, abbreviations, and aliases.
    Returns (is_valid, canonical_name).
    """
    if not feature_name:
        return False, None

    cleaned = str(feature_name).strip().lower().replace(" ", "_")

    # 1. Exact match against SELECTED_FEATURES
    for valid_feat in SELECTED_FEATURES:
        if valid_feat.lower() == cleaned or valid_feat.lower().replace("_", "") == cleaned.replace("_", ""):
            return True, valid_feat

    # 2. Check alias dictionary
    alias_key = str(feature_name).strip().lower().replace("_", " ")
    if alias_key in _FEATURE_NAME_ALIASES:
        canonical = _FEATURE_NAME_ALIASES[alias_key]
        if canonical in SELECTED_FEATURES:
            return True, canonical

    # 3. Fuzzy match against SELECTED_FEATURES and aliases (strict cutoff to avoid generic 'condition' false positives)
    candidates = list(SELECTED_FEATURES) + list(_FEATURE_NAME_ALIASES.keys())
    matches = difflib.get_close_matches(alias_key, candidates, n=1, cutoff=0.85)
    if matches:
        matched = matches[0]
        canonical = _FEATURE_NAME_ALIASES.get(matched, matched)
        if canonical in SELECTED_FEATURES:
            return True, canonical

    return False, None



def fuzzy_match_categorical(canonical_name: str, raw_value: Any) -> Optional[str]:
    """
    Match raw_value against allowed values for a categorical feature with spelling tolerance.
    """
    if raw_value is None or canonical_name not in CATEGORICAL_VALUES:
        return None

    val_str = str(raw_value).strip().lower()
    allowed_values = CATEGORICAL_VALUES[canonical_name]

    # 1. Exact case-insensitive match
    for allowed in allowed_values:
        if allowed.lower() == val_str:
            return allowed

    # 2. Synonyms and common typo dictionary match
    synonyms = _CATEGORICAL_SYNONYMS.get(canonical_name, {})
    if val_str in synonyms:
        return synonyms[val_str]

    for syn, canonical_val in synonyms.items():
        if syn in val_str or val_str in syn:
            return canonical_val

    # 3. Substring match against allowed values
    for allowed in allowed_values:
        if allowed.lower() in val_str or val_str in allowed.lower():
            return allowed

    # 4. Fuzzy string matching against allowed values + synonyms
    all_keys = list(synonyms.keys()) + [a.lower() for a in allowed_values]
    matches = difflib.get_close_matches(val_str, all_keys, n=1, cutoff=0.65)
    if matches:
        matched_key = matches[0]
        if matched_key in synonyms:
            return synonyms[matched_key]
        for allowed in allowed_values:
            if allowed.lower() == matched_key:
                return allowed

    return None


def validate_feature_value(feature_name: str, raw_value: Any) -> Tuple[bool, Any]:
    """
    Validate and normalize a value for a specific schema feature.
    - Numerical: non-negative integer or float
    - Date: ISO-8601 string (YYYY-MM-DD)
    - Categorical: valid dataset category
    """
    is_valid_feat, canonical_name = validate_feature_name(feature_name)
    if not is_valid_feat or canonical_name is None:
        return False, None

    if raw_value is None:
        return False, None

    # Handle numerical features
    if canonical_name in NUMERICAL_FEATURES:
        try:
            if isinstance(raw_value, (int, float)):
                num_val = float(raw_value)
            else:
                val_str = str(raw_value).strip().lower()
                # Check number words
                if val_str in _NUMBER_WORDS:
                    num_val = float(_NUMBER_WORDS[val_str])
                else:
                    nums = re.findall(r"\d+(?:\.\d+)?", val_str)
                    if not nums:
                        return False, None
                    num_val = float(nums[0])

            if num_val < 0:
                return False, None

            final_val = int(num_val) if num_val.is_integer() else num_val
            return True, final_val
        except (ValueError, TypeError):
            return False, None

    # Handle date features
    if canonical_name in DATE_FEATURES:
        value = normalise_date_value(raw_value)
        if value:
            return True, value
        return False, None

    # Handle categorical features
    if canonical_name in CATEGORICAL_FEATURES:
        matched = fuzzy_match_categorical(canonical_name, raw_value)
        if matched:
            return True, matched
        return False, None

    return False, None


def extract_explicit_facts(user_message: str, current_question_feature: Optional[str] = None) -> Dict[str, Any]:
    """
    Deterministically extract facts from user message and active question context.
    Supports short contextual answers, natural language phrasing, spelling errors,
    dates, and multiple facts in a single turn.
    """
    text = (user_message or "").strip()
    lower = text.lower()
    result: Dict[str, Any] = {
        "extracted_features": [],
        "unavailable_features": [],
        "raw_observations": [],
        "user_intent": "inform",
    }
    if not text:
        return result

    # Check for direct uncertainty ("I don't know", "not sure")
    if current_question_feature and any(token in lower for token in _UNKNOWN_PATTERNS):
        is_valid_f, canon_f = validate_feature_name(current_question_feature)
        if is_valid_f and canon_f:
            result["unavailable_features"].append(canon_f)
            result["user_intent"] = "decline"
            return result

    def add(name: str, value: Any) -> None:
        is_valid_f, canon_f = validate_feature_name(name)
        if is_valid_f and canon_f:
            is_valid_v, canon_v = validate_feature_value(canon_f, value)
            if is_valid_v and canon_v is not None:
                if not any(item["name"] == canon_f for item in result["extracted_features"]):
                    result["extracted_features"].append({"name": canon_f, "value": canon_v})

    # -------------------------------------------------------------------------
    # 1. Context-Aware Interpretation (Active Question Context)
    # -------------------------------------------------------------------------
    if current_question_feature:
        is_valid_curr, curr_canonical = validate_feature_name(current_question_feature)
        if is_valid_curr and curr_canonical:
            # Check numerical active question (e.g. "2", "2 years", "3", "3 problems", "4 services", "four")
            if curr_canonical in NUMERICAL_FEATURES:
                # Check if the response mentions a unit specifically belonging to a different feature
                has_other_unit = (
                    ("problem" in lower or "issue" in lower) and curr_canonical != "Reported_Issues"
                ) or (
                    ("year" in lower or "yr" in lower) and curr_canonical != "Vehicle_Age"
                ) or (
                    ("service" in lower or "visit" in lower) and curr_canonical != "Service_History"
                ) or (
                    ("accident" in lower or "crash" in lower) and curr_canonical != "Accident_History"
                ) or (
                    "cc" in lower and curr_canonical != "Engine_Size"
                ) or (
                    ("mile" in lower or "km" in lower) and curr_canonical not in ("Mileage", "Odometer_Reading")
                )
                if not has_other_unit:
                    # Bare number or matching unit
                    num_match = re.search(r"^(?:age\s*(?:is|=|:)?\s*|it\s*is\s*|is\s*|around\s*|about\s*)?(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)(?:\s*(?:years?|yrs?|problems?|issues?|services?|visits?|times?|accidents?|cc|km|miles))?\.?$", lower)
                    if num_match:
                        raw_n = num_match.group(1)
                        val = _NUMBER_WORDS.get(raw_n, raw_n)
                        add(curr_canonical, val)


            # Check categorical active question (e.g. "good", "good brakes", "weak", "batteru is weak", "average", "maintainance is avrage")
            elif curr_canonical in CATEGORICAL_FEATURES:
                cat_val = fuzzy_match_categorical(curr_canonical, text)
                if cat_val:
                    add(curr_canonical, cat_val)

            # Check date active question (e.g. "27/08/2025", "15 March 2025")
            elif curr_canonical in DATE_FEATURES:
                date_val = normalise_date_value(text)
                if date_val:
                    add(curr_canonical, date_val)

    # -------------------------------------------------------------------------
    # 2. Multi-Fact & Standalone Fact Patterns
    # -------------------------------------------------------------------------
    # Age: "2 years", "5 years old", "age is 2 years", "age 2", "vehicle is 5 years old"
    age_match = re.search(r"\b(?:age\s*(?:is|=|:)?\s*|vehicle\s+age\s*(?:is|=|:)?\s*|car\s+is\s*|vehicle\s+is\s*)?(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*(?:years?|yrs?)(?:\s+old)?\b|\bage\s*(?:is|=|:)\s*(\d+(?:\.\d+)?)\b", lower)
    if age_match:
        val = next(g for g in age_match.groups() if g is not None)
        add("Vehicle_Age", _NUMBER_WORDS.get(val, val))

    # Reported issues: "3 problems", "3 reported issues", "3 issues", "has 3 problems"
    issue_match = re.search(r"\b(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*(?:reported\s+)?(?:issues?|problems?|faults?)\b|\b(?:reported\s+)?(?:issues?|problems?)\s*(?:is|are|=|:)\s*(\d+(?:\.\d+)?)\b", lower)
    if issue_match:
        val = next(g for g in issue_match.groups() if g is not None)
        add("Reported_Issues", _NUMBER_WORDS.get(val, val))

    # Service history: "4 services", "serviced 3 times", "servied 3 times", "4 past services", "3 service visits"
    service_match = re.search(
        r"\b(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*(?:past\s+)?services?\b|"
        r"\b(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*(?:past\s+)?service\s*(?:visits?|appointments?)\b|"
        r"\b(?:serviced|servied|service)\s+(?:it|the\s+vehicle)?\s*(?:for\s+)?(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*times\b|"
        r"\bservice\s*history\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)\b",
        lower
    )
    if service_match:
        val = next(g for g in service_match.groups() if g is not None)
        add("Service_History", _NUMBER_WORDS.get(val, val))

    # Accident history: "0 accidents", "no accidents", "never had an accident", "2 accidents"
    accident_match = re.search(r"\b(\d+(?:\.\d+)?|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b)\s*(?:past\s+)?(?:accidents?|crashes?)\b|\bno\s+accidents?\b|\bnever\s+(?:had\s+an\s+|been\s+in\s+an\s+)?accidents?\b|\baccident\s*history\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)\b", lower)
    if accident_match:
        if "no accidents" in lower or "never" in lower:
            add("Accident_History", 0)
        else:
            val = next(g for g in accident_match.groups() if g is not None)
            add("Accident_History", _NUMBER_WORDS.get(val, val))

    # Mileage & Odometer
    mileage_match = re.search(r"\bmileage\s*(?:is|=|:)?\s*([\d,]+(?:\.\d+)?)\b", lower)
    if mileage_match:
        add("Mileage", mileage_match.group(1).replace(",", ""))

    odometer_match = re.search(r"\bodometer(?:\s+reading)?\s*(?:is|=|:)?\s*([\d,]+(?:\.\d+)?)\b", lower)
    if odometer_match:
        add("Odometer_Reading", odometer_match.group(1).replace(",", ""))

    # Engine size
    engine_match = re.search(r"\b(?:engine\s*(?:size)?\s*(?:is|=|:)?\s*)?(\d+(?:\.\d+)?)\s*cc\b", lower)
    if engine_match:
        add("Engine_Size", engine_match.group(1))

    # Brake condition: "good brakes", "brakes are good", "brke is good", "breaks are good", "brakes worn out"
    if any(k in lower for k in ("brake", "brakes", "brke", "breaks", "break")):
        cat_val = fuzzy_match_categorical("Brake_Condition", lower)
        if cat_val:
            add("Brake_Condition", cat_val)

    # Battery status: "battery is good", "good battery", "batteru is weak", "batery is weak", "battery weak", "battery is wek"
    if any(k in lower for k in ("battery", "batteru", "batery")):
        cat_val = fuzzy_match_categorical("Battery_Status", lower)
        if cat_val:
            add("Battery_Status", cat_val)

    # Maintenance history: "maintenance is average", "maintainance is avrage", "maintenence average", "maintenance avg"
    if any(k in lower for k in ("maintenance", "maintainance", "maintenence", "maintanence")):
        cat_val = fuzzy_match_categorical("Maintenance_History", lower)
        if cat_val:
            add("Maintenance_History", cat_val)

    # Tire condition: "tire condition is good", "tires are worn out", "good tires"
    if any(k in lower for k in ("tire", "tires", "tyre", "tyres")):
        cat_val = fuzzy_match_categorical("Tire_Condition", lower)
        if cat_val:
            add("Tire_Condition", cat_val)

    # Date extraction: "last serviced on 27 august 2025", "last service was 27/08/2025"
    date_val = normalise_date_value(text)
    if date_val:
        if any(k in lower for k in ("last service", "last serviced", "serviced on", "serviced", "service was", "service date")):
            add("Last_Service_Date", date_val)
        elif any(k in lower for k in ("warranty", "warranty expiry")):
            add("Warranty_Expiry_Date", date_val)

    # Intent detection
    if "brake" in lower:
        result["user_intent"] = "brake"
    elif "battery" in lower:
        result["user_intent"] = "battery"
    elif "tire" in lower or "tyre" in lower:
        result["user_intent"] = "tire"
    elif any(p in lower for p in ("won't start", "wont start", "doesn't start", "doesnt start", "starting")):
        result["user_intent"] = "starting"
    elif "overheat" in lower:
        result["user_intent"] = "overheating"
    elif "accident" in lower or "crash" in lower:
        result["user_intent"] = "accident"
    elif "service" in lower or "maintenance" in lower:
        result["user_intent"] = "maintenance_history"

    if not result["extracted_features"] and not result["unavailable_features"]:
        result["raw_observations"].append(text)

    return result


def constrain_extraction_to_explicit_facts(
    extraction: Dict[str, Any], user_message: str, current_question_feature: Optional[str]
) -> Dict[str, Any]:
    """
    Intersect and merge InputAgent LLM extraction with deterministic extraction.
    Guarantees strict schema adherence while supporting natural language understanding.
    """
    evidenced = extract_explicit_facts(user_message, current_question_feature)

    # Process InputAgent extractions
    llm_extracted = extraction.get("extracted_features", []) if isinstance(extraction, dict) else []
    if isinstance(llm_extracted, list):
        for item in llm_extracted:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            val = item.get("value")

            is_valid_f, canonical_f = validate_feature_name(name)
            if not is_valid_f or canonical_f is None:
                continue

            is_valid_v, canonical_v = validate_feature_value(canonical_f, val)
            if not is_valid_v or canonical_v is None:
                continue

            # Add if not already present
            if not any(f["name"] == canonical_f for f in evidenced["extracted_features"]):
                evidenced["extracted_features"].append({"name": canonical_f, "value": canonical_v})

    # Process unavailable features
    llm_unavailable = extraction.get("unavailable_features", []) if isinstance(extraction, dict) else []
    if isinstance(llm_unavailable, list):
        for name in llm_unavailable:
            is_valid_f, canonical_f = validate_feature_name(name)
            if is_valid_f and canonical_f and canonical_f not in evidenced["unavailable_features"]:
                evidenced["unavailable_features"].append(canonical_f)

    # Preserve raw observations
    llm_obs = extraction.get("raw_observations", []) if isinstance(extraction, dict) else []
    if isinstance(llm_obs, list):
        for obs in llm_obs:
            if obs and isinstance(obs, str) and obs not in evidenced["raw_observations"]:
                evidenced["raw_observations"].append(obs)

    # User intent
    if isinstance(extraction, dict) and extraction.get("user_intent"):
        evidenced["user_intent"] = extraction["user_intent"]

    return evidenced


def apply_structured_extraction(
    state: AgentState,
    extraction: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate LLM extraction and update AgentState deterministically.
    """
    state_changed = False
    updated_features: List[str] = []
    declined_features: List[str] = []
    rejected_features: List[str] = []

    # 1. Process explicit feature extractions
    raw_extracted = extraction.get("extracted_features", [])
    if isinstance(raw_extracted, list):
        for item in raw_extracted:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            val = item.get("value")

            is_valid_feat, canonical_name = validate_feature_name(name)
            if not is_valid_feat or canonical_name is None:
                logger.warning(f"[FEATURE_TOOL] Rejected non-schema feature: '{name}'")
                rejected_features.append(name)
                continue

            is_valid_val, canonical_val = validate_feature_value(canonical_name, val)
            if not is_valid_val or canonical_val is None:
                logger.warning(f"[FEATURE_TOOL] Rejected invalid value for '{canonical_name}': {val}")
                rejected_features.append(f"{canonical_name}:{val}")
                continue

            # Update feature in state
            changed = state.update_feature(canonical_name, canonical_val, status="known")
            if changed:
                state_changed = True
                updated_features.append(canonical_name)
                logger.info(f"[STATE] Confirmed {canonical_name} = {canonical_val}")

    # 2. Process unavailable features ("I don't know")
    raw_unavailable = extraction.get("unavailable_features", [])
    if isinstance(raw_unavailable, list):
        for name in raw_unavailable:
            is_valid_feat, canonical_name = validate_feature_name(name)
            if is_valid_feat and canonical_name:
                state.mark_declined(canonical_name)
                declined_features.append(canonical_name)
                state_changed = True
                logger.info(f"[STATE] Marked {canonical_name} as unavailable (user declined)")

    # 3. Store raw observations
    raw_obs = extraction.get("raw_observations", [])
    if isinstance(raw_obs, list):
        for obs in raw_obs:
            if obs and isinstance(obs, str) and obs not in state.raw_observations:
                state.raw_observations.append(obs)

    return {
        "state_changed": state_changed,
        "updated_features": updated_features,
        "declined_features": declined_features,
        "rejected_features": rejected_features,
        "known_count": state.known_feature_count(),
    }
