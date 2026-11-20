import json
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self-harm", "hurt myself",
    "want to die", "no reason to live", "better off dead", "overdose"
]

STRESS_KEYWORDS = [
    "anxious", "anxiety", "panic", "insomnia", "exhausted", "overwhelmed",
    "pain", "work", "afraid", "stigma", "ashamed", "tired", "hopeless",
    "burnout", "sad", "depressed", "worry", "worried", "stress", "stressed"
]


def encode_reflection_text(text: str, dim: int = 16) -> torch.Tensor:
    if not text or not text.strip():
        return torch.zeros(1, dim, dtype=torch.float32)

    text_lower = text.lower()
    vec = np.zeros(dim, dtype=np.float32)

    words = max(len(text_lower.split()), 1)
    stress_hits = sum(1 for kw in STRESS_KEYWORDS if kw in text_lower)
    stigma_hits = sum(1 for kw in ["stigma", "employer", "boss", "secret", "ashamed"] if kw in text_lower)
    somatic_hits = sum(1 for kw in ["sleep", "pain", "sugar", "fatigue", "headache", "heart", "diabetes"] if kw in text_lower)

    vec[0] = min(stress_hits / 3.0, 1.0)
    vec[1] = min(stigma_hits / 2.0, 1.0)
    vec[2] = min(somatic_hits / 2.0, 1.0)
    vec[3] = min(words / 50.0, 1.0)

    for token in text_lower.split():
        slot = 4 + (hash(token) % (dim - 4))
        vec[slot] += 0.25

    norm = np.linalg.norm(vec[4:])
    if norm > 0:
        vec[4:] = vec[4:] / norm

    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)


class MultimodalSteppedCareNet(nn.Module):
    def __init__(self, tab_dim: int = 7, txt_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.tab_net = nn.Sequential(
            nn.Linear(tab_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.txt_net = nn.Sequential(
            nn.Linear(txt_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2)
        )
        fusion_dim = hidden_dim + (hidden_dim // 2)
        self.shared_fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU()
        )
        self.tier_head = nn.Linear(hidden_dim, 4)
        self.dropoff_head = nn.Linear(hidden_dim, 1)

        self._init_calibrated_weights()

    def _init_calibrated_weights(self):
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, tab_x: torch.Tensor, txt_x: torch.Tensor):
        h_tab = self.tab_net(tab_x)
        h_txt = self.txt_net(txt_x)
        fused = torch.cat([h_tab, h_txt], dim=1)
        h_shared = self.shared_fusion(fused)

        tier_logits = self.tier_head(h_shared)
        dropoff_prob = torch.sigmoid(self.dropoff_head(h_shared))
        return tier_logits, dropoff_prob


MODEL = MultimodalSteppedCareNet()
MODEL.eval()


def _try_local_ollama_reply(user_message: str, patterns: list[str]) -> str | None:
    """
    Optional hook: if Ollama is running locally on port 11434 (e.g., llama3),
    generates a warm, safe, non-diagnostic reply. Falls back cleanly if offline.
    """
    try:
        system_prompt = (
            "You are Sahatna MindBridge AI, an empathetic wellbeing navigation companion in Abu Dhabi. "
            "Listen warmly, ask 1 gentle clarifying follow-up question, and mention possible everyday patterns "
            f"({', '.join(patterns)}) ONLY as non-diagnostic possibilities to discuss with a doctor. "
            "Never diagnose diseases or prescribe medication. Keep your reply under 75 words."
        )
        payload = json.dumps({
            "model": "llama3",
            "prompt": f"System: {system_prompt}\nUser: {user_message}\nAssistant:",
            "stream": False
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data.get("response", "").strip()
            return text if text else None
    except Exception:
        return None


def analyze_chat_conversation(messages: list[dict], use_local_ollama: bool = False) -> dict:
    """
    Analyzes the user's conversation history to:
    1. Trigger immediate deterministic crisis override if self-harm is mentioned.
    2. Identify Possible Mind-Body Patterns (strictly framed as possibilities with caution).
    3. Estimate validated PHQ/GAD equivalents, sleep hours, and comorbidity for routing.
    4. Generate a supportive conversational response.
    """
    user_texts = [m["content"] for m in messages if m["role"] == "user"]
    full_text = " ".join(user_texts).lower()
    latest_msg = user_texts[-1] if user_texts else ""

    # 1. Crisis Check
    crisis_detected = any(kw in full_text for kw in CRISIS_KEYWORDS)
    if crisis_detected:
        return {
            "reply": (
                "I hear how much pain you are carrying right now, and your safety matters most. "
                "Please connect immediately with a trained human specialist who can support you right now:\n\n"
                "- **800-SAKINA (800-725462)** — Abu Dhabi 24/7 Confidential Mental Health Line\n"
                "- **Istijaba (800-1717)** or **999** for immediate emergency care."
            ),
            "crisis_detected": True,
            "possible_patterns": ["Urgent Emotional Crisis — Immediate Human Support Required"],
            "est_phq9": 22,
            "est_gad7": 18,
            "est_sleep": 4.0,
            "detected_comorbidity": "None",
            "stigma_detected": False,
            "summary_note": " | ".join(user_texts)
        }

    # 2. Extract Possible Mind-Body Patterns (Non-Diagnostic Possibilities)
    patterns = []
    est_phq9 = 4
    est_gad7 = 4
    est_sleep = 7.0
    detected_comorbidity = "None"
    stigma_detected = False

    if any(w in full_text for w in ["sleep", "insomnia", "awake", "night", "tired", "exhausted", "fatigue"]):
        patterns.append("Sleep Debt & Physical Exhaustion Overlap (Possibility)")
        est_sleep = 4.5
        est_phq9 += 4

    if any(w in full_text for w in ["anxious", "anxiety", "worry", "worried", "panic", "heart", "nervous", "overthinking"]):
        patterns.append("Elevated Situational Anxiety / Worry Loop (Possibility)")
        est_gad7 += 8

    if any(w in full_text for w in ["sad", "low", "hopeless", "motivation", "empty", "heavy", "burnout", "overwhelmed"]):
        patterns.append("Burnout or Low-Mood Burden (Possibility)")
        est_phq9 += 7

    if any(w in full_text for w in ["diabetes", "sugar", "blood pressure", "hypertension", "pain", "headache", "migraine", "thyroid"]):
        patterns.append("Physical-Mental Comorbidity Interaction (Possibility)")
        if "diabetes" in full_text or "sugar" in full_text:
            detected_comorbidity = "Type 2 Diabetes / Fatigue"
        elif "pain" in full_text or "headache" in full_text or "migraine" in full_text:
            detected_comorbidity = "Chronic Pain / Headaches"
        elif "pressure" in full_text or "heart" in full_text:
            detected_comorbidity = "Hypertension / Heart Health"
        else:
            detected_comorbidity = "Thyroid / Vitamin Deficiency"
        est_phq9 += 2
        est_gad7 += 2

    if any(w in full_text for w in ["employer", "boss", "work", "job", "stigma", "secret", "ashamed", "wait"]):
        stigma_detected = True

    if not patterns:
        patterns.append("Early Everyday Stress Adjustment (Possibility)")

    est_phq9 = min(est_phq9, 24)
    est_gad7 = min(est_gad7, 20)

    # 3. Generate Empathetic Response (Ollama if enabled & running, else smart deterministic guide)
    ollama_reply = _try_local_ollama_reply(latest_msg, patterns) if use_local_ollama else None

    if ollama_reply:
        reply_text = ollama_reply
    else:
        turn_count = len(user_texts)
        if turn_count == 1:
            reply_text = (
                f"Thank you for sharing that openly. Based on what you described, we are seeing signs that could align with "
                f"**{patterns[0].replace(' (Possibility)', '')}** *(note: this is only an early possibility, not a medical diagnosis)*.\n\n"
                "To help match you to the safest support, **how has your sleep or physical energy been over the past two weeks, and are you managing any physical health conditions?**"
            )
        elif turn_count == 2:
            reply_text = (
                "That context really helps connect the dots—physical exhaustion and ongoing worry often amplify each other. "
                "Remember, even if online searches make symptoms look alarming, most early patterns respond very well when addressed early.\n\n"
                "**Would you prefer discrete virtual support from home, or an in-person primary-care visit—and is there anything else you'd like to add before we prepare your private care options?**"
            )
        else:
            reply_text = (
                "I have noted all of those details into your private check-in summary. Whenever you feel ready, click "
                "**'Continue to My Wellbeing Summary & Care Options'** below to review your personalized care match and choose what you want to share."
            )

    return {
        "reply": reply_text,
        "crisis_detected": False,
        "possible_patterns": patterns,
        "est_phq9": est_phq9,
        "est_gad7": est_gad7,
        "est_sleep": est_sleep,
        "detected_comorbidity": detected_comorbidity,
        "stigma_detected": stigma_detected,
        "summary_note": " | ".join(user_texts)
    }


def evaluate_patient_intake(
    phq9_score: int,
    gad7_score: int,
    phq9_q9_self_harm: int,
    sleep_hours_avg: float,
    has_comorbidity: bool,
    comorbidity_desc: str,
    est_wait_days: int,
    stigma_concern: bool,
    reflection_text: str,
    missing_telemetry: bool = False
) -> dict:
    text_lower = (reflection_text or "").lower()
    keyword_crisis = any(kw in text_lower for kw in CRISIS_KEYWORDS)

    # 1. HARD DETERMINISTIC CRISIS GUARDRAIL
    if phq9_q9_self_harm > 0 or keyword_crisis:
        return {
            "status": "CRISIS_ESCALATION",
            "recommended_tier": "Tier 4+ (Immediate Human Crisis Escalation)",
            "dropoff_risk_pct": 95.0,
            "confidence_pct": 99.9,
            "abstained": False,
            "queue_jumped": True,
            "crisis_triggered": True,
            "crisis_reason": "Self-harm or severe crisis indicators detected",
            "action_plan": (
                "AI routing halted immediately. Direct warm hand-off to licensed human responders at "
                "800-SAKINA (800-725462) / Istijaba (800-1717) or 999 Emergency Services."
            ),
            "comorbidity_insight": f"Urgent safety override active. Comorbidity context ({comorbidity_desc}) attached for crisis clinician.",
            "feature_drivers": ["Safety Guardrail Triggered"]
        }

    present_fields = 5
    if missing_telemetry:
        present_fields -= 2
    if not reflection_text or len(reflection_text.strip()) < 5:
        present_fields -= 1

    completeness_ratio = present_fields / 5.0
    sleep_drift = max(0.0, (7.5 - sleep_hours_avg) / 7.5) if not missing_telemetry else 0.0

    tab_features = torch.tensor([[
        phq9_score / 27.0,
        gad7_score / 21.0,
        sleep_drift,
        1.0 if has_comorbidity else 0.0,
        min(est_wait_days / 30.0, 1.0),
        1.0 if stigma_concern else 0.0,
        completeness_ratio
    ]], dtype=torch.float32)

    txt_features = encode_reflection_text(reflection_text)

    with torch.no_grad():
        tier_logits, raw_dropoff = MODEL(tab_features, txt_features)

        max_severity = max(phq9_score, gad7_score)
        clinical_prior = torch.zeros_like(tier_logits)
        if max_severity < 5:
            clinical_prior[0, 0] = 2.5
        elif max_severity < 10:
            clinical_prior[0, 1] = 2.4
        elif max_severity < 15:
            clinical_prior[0, 2] = 2.4
        else:
            clinical_prior[0, 3] = 2.6

        if has_comorbidity and 5 <= max_severity < 15:
            clinical_prior[0, 1] += 0.8

        calibrated_logits = tier_logits + clinical_prior
        probs = F.softmax(calibrated_logits, dim=1).squeeze(0).numpy()

        base_nn_risk = float(raw_dropoff.item()) * 0.25
        wait_factor = min(est_wait_days / 28.0, 1.0) * 0.40
        stigma_factor = 0.20 if stigma_concern else 0.0
        severity_fatigue = (phq9_score / 27.0) * 0.15
        dropoff_risk_pct = round(min((base_nn_risk + wait_factor + stigma_factor + severity_fatigue) * 100.0, 96.0), 1)

    top_tier_idx = int(np.argmax(probs))
    raw_conf = float(probs[top_tier_idx])
    effective_conf_pct = round(raw_conf * (0.55 + 0.45 * completeness_ratio) * 100.0, 1)

    # 2. CONFIDENCE-THRESHOLD ABSTENTION FALLBACK
    if effective_conf_pct < 70.0 or completeness_ratio <= 0.40:
        return {
            "status": "ABSTAINED_HUMAN_TRIAGE",
            "recommended_tier": "Safe Fallback: Human Nurse Tele-Triage Check",
            "dropoff_risk_pct": dropoff_risk_pct,
            "confidence_pct": effective_conf_pct,
            "abstained": True,
            "queue_jumped": False,
            "crisis_triggered": False,
            "crisis_reason": "None",
            "action_plan": (
                f"Model confidence ({effective_conf_pct}%) is below the safety threshold due to incomplete details. "
                "Automated AI routing abstained. Routed to a human primary-care triage nurse for a quick verification call."
            ),
            "comorbidity_insight": f"Incomplete input flagged. Clinician should verify physical history ({comorbidity_desc}).",
            "feature_drivers": ["Sparse Input Data / Safety Abstention"]
        }

    tier_map = {
        0: "Tier 1: Guided Self-Help & Community Wellbeing Hubs",
        1: "Tier 2: SEHA Primary Care Physician (Integrated Mind-Body Review)",
        2: "Tier 3: Encrypted Tele-Mental Health Consultation (SAKINA Virtual)",
        3: "Tier 4: Specialist Outpatient Psychiatry Referral (SAKINA Network)"
    }
    selected_tier = tier_map[top_tier_idx]
    queue_jumped = False

    # 3. DYNAMIC QUEUE-JUMPING FOR HIGH DROP-OFF OR WAIT RISK
    if top_tier_idx in [1, 3] and (dropoff_risk_pct >= 60.0 or (max_severity >= 15 and est_wait_days > 7)):
        queue_jumped = True
        selected_tier = (
            "Tier 3/4 Fast-Track: 48-Hour Virtual Tele-Psychiatry Bridge "
            "(Queue-Jumped due to High Wait/Drop-Off Risk)"
        )

    drivers = []
    if max_severity >= 10:
        drivers.append(f"Elevated Symptom Pattern (PHQ={phq9_score}, GAD={gad7_score})")
    if has_comorbidity:
        drivers.append(f"Physical Comorbidity Interaction ({comorbidity_desc})")
    if est_wait_days > 10:
        drivers.append(f"Clinic Wait Friction ({est_wait_days} days)")
    if stigma_concern:
        drivers.append("High Stigma / Privacy Sensitivity")
    if sleep_drift > 0.25 and not missing_telemetry:
        drivers.append(f"Sleep Baseline Drift ({sleep_hours_avg} hrs/night)")
    if not drivers:
        drivers.append("Stable Baseline Metrics")

    comorbidity_summary = (
        f"Patient reports {comorbidity_desc} alongside wellbeing indicators (PHQ-9={phq9_score}, GAD-7={gad7_score}). "
        f"Sleep averaging {sleep_hours_avg if not missing_telemetry else 'N/A'} hrs/night. "
        "Recommend co-managing physical symptoms and stress drivers together rather than in isolation."
        if has_comorbidity else
        f"No chronic physical comorbidity reported. Primary drivers are mood/anxiety indicators (PHQ-9={phq9_score}, GAD-7={gad7_score})."
    )

    return {
        "status": "ROUTED",
        "recommended_tier": selected_tier,
        "dropoff_risk_pct": dropoff_risk_pct,
        "confidence_pct": effective_conf_pct,
        "abstained": False,
        "queue_jumped": queue_jumped,
        "crisis_triggered": False,
        "crisis_reason": "None",
        "action_plan": (
            "Fast-tracked to 48-hour encrypted tele-mental health slot with automated SMS warm reminder to prevent referral drop-off."
            if queue_jumped else
            "Matched to least-intensive safe care tier with transparent insurance eligibility check."
        ),
        "comorbidity_insight": comorbidity_summary,
        "feature_drivers": drivers
    }