"""
Lunar AI -- confidence-aware scientific interpretation module.

This is deliberately NOT a general-purpose chatbot. It is a structured
analysis engine that:
  1. Only ever describes what the registration pipeline actually
     measured (feature counts, inlier ratios, RMSE, coverage, per-sensor
     structural contribution maps).
  2. Classifies every statement as OBSERVED (directly read off the
     data), DERIVED (a reasonable interpretation of measured evidence),
     or UNCERTAIN (cannot be determined from available data).
  3. Explicitly refuses to state geolocation, mineral composition,
     geological age, or other information the pipeline has no means of
     measuring -- these are never available in this system and are
     always reported as UNCERTAIN / not determinable.
  4. Scales its confidence and the strength of its language to the
     registration confidence of each sensor pair.

An optional enhancement step can call the Anthropic API (if
ANTHROPIC_API_KEY is set in the environment) to turn the structured
findings below into fluent prose, WITH THE MODEL EXPLICITLY CONSTRAINED
to only rephrase supplied findings, not invent new ones. If no key is
configured, or the call fails for any reason, the deterministic
rule-based narrative below is used as-is -- the system never blocks or
degrades because an optional LLM call isn't available.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

try:
    import urllib.request
    import urllib.error
except ImportError:  # pragma: no cover
    urllib = None


def _feature_observations(sensor: str, reg: dict, info_pres: dict) -> List[Dict]:
    obs = []
    if reg.get("status") != "SUCCESS":
        obs.append({
            "type": "UNCERTAIN",
            "text": f"{sensor} could not be geometrically registered to OHRC "
                    f"({reg.get('failure_reason', 'insufficient correspondences')}). "
                    "No cross-sensor structural comparison is possible for this sensor in this run.",
        })
        return obs

    obs.append({
        "type": "OBSERVED",
        "text": (f"{sensor} was registered to OHRC using a "
                  f"{reg.get('transformation_type')} estimated from "
                  f"{reg.get('inlier_matches')} inlier correspondences out of "
                  f"{reg.get('valid_matches')} candidate matches "
                  f"(inlier ratio {reg.get('inlier_ratio', 0)*100:.1f}%, "
                  f"reprojection RMSE {reg.get('rmse_px')} px)."),
    })

    conf = reg.get("confidence")
    if conf == "HIGH":
        obs.append({
            "type": "DERIVED",
            "text": f"Registration confidence for {sensor} is HIGH; spatial correspondence "
                     "between OHRC and this sensor's structural content is likely reliable "
                     "across most of the overlapping frame.",
        })
    elif conf == "MEDIUM":
        obs.append({
            "type": "DERIVED",
            "text": f"Registration confidence for {sensor} is MEDIUM; correspondence appears "
                     "usable but should be treated with some caution, particularly away from "
                     "the detected inlier locations.",
        })
    else:
        obs.append({
            "type": "UNCERTAIN",
            "text": f"Registration confidence for {sensor} is LOW. Any apparent cross-sensor "
                     "structural agreement should be treated as unreliable pending a stronger "
                     "correspondence set.",
        })

    ip = info_pres or {}
    if ip.get("issues"):
        for issue in ip["issues"]:
            obs.append({"type": "OBSERVED", "text": f"{sensor}: {issue}"})
    else:
        obs.append({
            "type": "OBSERVED",
            "text": f"{sensor}: no information-preservation issues detected "
                     f"(warped coverage {ip.get('coverage_fraction', 0)*100:.1f}% of the reference frame).",
        })

    return obs


def build_analysis(sensor_results: Dict[str, dict]) -> Dict:
    """
    sensor_results: {sensor: registration_result_dict (with information_preservation)}
    """
    observations: List[Dict] = []
    confidences = []

    for sensor, reg in sensor_results.items():
        info_pres = reg.get("information_preservation")
        observations.extend(_feature_observations(sensor, reg, info_pres))
        if reg.get("status") == "SUCCESS" and reg.get("confidence"):
            confidences.append(reg["confidence"])

    if not confidences:
        overall = "LOW"
    elif all(c == "HIGH" for c in confidences):
        overall = "HIGH"
    elif any(c == "LOW" for c in confidences):
        overall = "LOW" if confidences.count("LOW") >= len(confidences) / 2 else "MEDIUM"
    else:
        overall = "MEDIUM"

    sensor_roles = {
        "OHRC": "High-resolution optical/spatial reference -- defines the geometric frame all "
                "other sensors are aligned to.",
        "TMC": "Terrain/mapping-oriented optical information; contributes broader-context "
               "surface structure.",
        "IIRS": "Infrared/spectral information; contributes potential compositional/thermal "
                "signal not visible to OHRC, though this system does not perform spectral "
                "unmixing or composition estimation.",
        "SAR": "Radar-based structural information; sensitive to surface roughness/geometry "
               "independent of illumination, contributing information OHRC cannot capture "
               "under poor lighting.",
    }

    limitations = [
        "This system performs geometric registration and structural comparison only. It does "
        "NOT perform photogrammetric geolocation, so no latitude/longitude/elevation values "
        "are produced -- any such claim would be fabricated and is deliberately withheld.",
        "IIRS spectral bands are not radiometrically unmixed here, so no mineral composition "
        "or material classification is claimed.",
        "No absolute geological age, temperature, or material-property estimate is possible "
        "from image registration alone.",
        "Feature labels such as 'possible crater-like structure' reflect morphological "
        "appearance only and are not validated ground-truth identifications.",
    ]

    if overall == "LOW":
        headline = ("Overall registration confidence is LOW for this run. Scientific "
                    "interpretation below should be treated as preliminary and cross-checked "
                    "against additional data before use.")
    elif overall == "MEDIUM":
        headline = ("Overall registration confidence is MEDIUM. The structural comparisons "
                    "below are usable for exploratory analysis but should not be treated as "
                    "final.")
    else:
        headline = ("Overall registration confidence is HIGH across the successfully "
                    "registered sensors, supporting reasonably confident cross-sensor "
                    "structural comparison.")

    recommended_next_steps = [
        "Visually inspect the inlier match overlay for each sensor pair before relying on "
        "the fused output.",
        "For any sensor with LOW confidence, consider re-acquiring imagery with better "
        "overlap/illumination match, or supplying additional control points.",
        "For scientific (as opposed to demonstration) use, validate structural findings "
        "against independently georeferenced lunar data products.",
    ]

    result = {
        "headline": headline,
        "overall_confidence": overall,
        "observations": observations,
        "sensor_roles": {s: sensor_roles.get(s, "") for s in sensor_results.keys() | {"OHRC"}},
        "limitations": limitations,
        "recommended_next_steps": recommended_next_steps,
        "narrative_source": "rule_based",
    }

    enhanced = _try_llm_enhance(result)
    if enhanced:
        result["narrative"] = enhanced
        result["narrative_source"] = "anthropic_api"
    else:
        result["narrative"] = headline

    return result


def _try_llm_enhance(structured: Dict) -> str | None:
    """
    Optional: use the Anthropic API to turn the structured, evidence-grounded
    findings into a short fluent scientific narrative, WITHOUT permitting the
    model to introduce any new claims. Silently returns None on any failure
    (no key configured, no network, API error) so the deterministic output
    above always remains a complete, correct fallback.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or urllib is None:
        return None

    obs_text = "\n".join(f"- [{o['type']}] {o['text']}" for o in structured["observations"])
    prompt = (
        "You are summarizing a lunar multi-sensor image registration report for a "
        "planetary scientist. Rephrase ONLY the findings below into a tight, "
        "well-organized 120-180 word scientific summary paragraph. Preserve every "
        "OBSERVED/DERIVED/UNCERTAIN distinction. Do NOT add any new fact, number, "
        "location, composition, or age that is not already present below.\n\n"
        f"Overall confidence: {structured['overall_confidence']}\n"
        f"Findings:\n{obs_text}\n\n"
        f"Limitations to mention briefly: {'; '.join(structured['limitations'])}\n"
    )

    body = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 400,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        parts = [b["text"] for b in payload.get("content", []) if b.get("type") == "text"]
        text = "\n".join(parts).strip()
        return text or None
    except Exception:
        return None
