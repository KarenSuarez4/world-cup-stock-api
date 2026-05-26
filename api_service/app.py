
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import HTTPException

BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = BASE_DIR.parent / "export"

with open(EXPORT_DIR / "feature_cols.json", "r", encoding="utf-8") as f:
    FEATURE_COLS = json.load(f)

with open(EXPORT_DIR / "model_meta.json", "r", encoding="utf-8") as f:
    MODEL_META = json.load(f)

MODEL = joblib.load(EXPORT_DIR / "model.joblib")

app = FastAPI(title="World Cup Stock API", version=MODEL_META.get("model_version", "v1"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TeamInput(BaseModel):
    id: str
    team: str
    country_code: Optional[str] = None
    flag_emoji: Optional[str] = None
    group: Optional[str] = None
    current_stock: Optional[int] = 0
    features: Dict[str, float] = Field(..., description="Features numericas del modelo")

class PredictionOut(BaseModel):
    id: str
    team: str
    country_code: Optional[str] = None
    flag_emoji: Optional[str] = None
    group: Optional[str] = None
    probability_advance_quarters: float
    probability_advance_semis: float
    probability_advance_final: float
    probability_champion: float
    current_stock: int
    recommendation: str
    recommendation_reason: str

class PredictionResponse(BaseModel):
    predictions: List[PredictionOut]
    last_updated: str
    model_version: str

def map_phase_probs(prob_top4: float):
    # Heuristica simple. Ajusta si tienes modelos separados por fase.
    p_quarters = max(0.0, min(1.0, prob_top4))
    p_semis = max(0.0, min(1.0, prob_top4 - 0.12))
    p_final = max(0.0, min(1.0, prob_top4 - 0.22))
    p_champ = max(0.0, min(1.0, prob_top4 - 0.32))
    return p_quarters, p_semis, p_final, p_champ

def build_recommendation(prob_quarters: float):
    if prob_quarters >= 0.5:
        return "increase_stock", "Probabilidad >= 50% de avanzar a cuartos."
    if prob_quarters >= 0.4:
        return "hold", "Probabilidad moderada. Mantener stock."
    return "promotion", "Probabilidad baja. Considerar promocion temprana."

def load_default_payload() -> list[TeamInput]:
    payload_path = EXPORT_DIR / "default_payload.json"
    if not payload_path.exists():
        return []
    raw = json.loads(payload_path.read_text(encoding="utf-8"))
    return [TeamInput(**item) for item in raw]

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_META.get("model_name") }
    

@app.post("/predictions", response_model=PredictionResponse)
def predict(payload: List[TeamInput]):
    if len(payload) == 0:
        payload = load_default_payload()
        if len(payload) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty payload. Provide data or upload default_payload.json"
            )

    rows = []
    for item in payload:
        row = {"team": item.team, **item.features}
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.reindex(columns=FEATURE_COLS)

    if hasattr(MODEL, "predict_proba"):
        probs = MODEL.predict_proba(df)[:, 1]
    else:
        preds = MODEL.predict(df)
        probs = preds.astype(float)

    predictions = []
    for idx, item in enumerate(payload):
        p_top4 = float(probs[idx])
        p_q, p_s, p_f, p_c = map_phase_probs(p_top4)
        rec, reason = build_recommendation(p_q)
        predictions.append({
            "id": item.id,
            "team": item.team,
            "country_code": item.country_code,
            "flag_emoji": item.flag_emoji,
            "group": item.group,
            "probability_advance_quarters": p_q,
            "probability_advance_semis": p_s,
            "probability_advance_final": p_f,
            "probability_champion": p_c,
            "current_stock": int(item.current_stock or 0),
            "recommendation": rec,
            "recommendation_reason": reason
        })

    return {
        "predictions": predictions,
        "last_updated": pd.Timestamp.utcnow().isoformat(),
        "model_version": MODEL_META.get("model_version", "v1")
    }
