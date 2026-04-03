# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # FastAPI Web Application
#
# Premium web dashboard for the Dynamic Rule Engine.
# Provides drag-and-drop dataset upload, auto-generated rule viewing,
# rule modification, and explainable anomaly results.

# %%
import os
import sys
import uuid
import json
import io
import traceback

import pandas as pd
import numpy as np

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# %%
# Add parent paths for imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
_engine_dir = os.path.dirname(_this_dir)
_models_dir = os.path.dirname(_engine_dir)
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)

from rule_engine.engine import RuleEngine, EngineResult
from rule_engine.rule_generator import Rule, RuleGenerator
from rule_engine.ml_integrations import run_ml_integrations

# %% [markdown]
# ## App Setup

# %%
app = FastAPI(
    title="Dynamic Rule Engine",
    description="Schema-agnostic anomaly detection with explainable rules",
    version="1.0.0",
)

# Static files & templates
app.mount("/static", StaticFiles(directory=os.path.join(_this_dir, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(_this_dir, "templates"))

# In-memory session store (for demo; use Redis/DB in production)
_sessions: dict[str, dict] = {}

# %% [markdown]
# ## Routes

# %%
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the main dashboard."""
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Accept a CSV/JSON file, run the full engine pipeline, and return results.
    """
    session_id = str(uuid.uuid4())[:8]

    try:
        contents = await file.read()
        filename = file.filename or "upload.csv"
        ext = os.path.splitext(filename)[1].lower()

        # Parse file
        if ext == ".json":
            df = pd.read_json(io.BytesIO(contents))
        else:
            df = pd.read_csv(io.BytesIO(contents))

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Limit preview for very large files
        row_limit = 100_000
        was_truncated = False
        if len(df) > row_limit:
            df = df.head(row_limit)
            was_truncated = True

        # Run engine
        engine = RuleEngine()
        result = engine.run(df)

        # Store session
        _sessions[session_id] = {
            "engine": engine,
            "result": result,
            "filename": filename,
            "was_truncated": was_truncated,
            "original_rows": len(df),
        }

        # Build response
        response = _build_response(session_id, result, filename, was_truncated)
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Engine error: {str(e)}")


@app.get("/api/results/{session_id}")
async def get_results(session_id: str, page: int = 1, page_size: int = 50,
                      anomalies_only: bool = False):
    """Get paginated results for a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    result = _sessions[session_id]["result"]
    df = result.result_df

    if anomalies_only:
        df = df[df["_is_anomaly"] == 1]

    total = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]

    rows = []
    for idx, row in page_df.iterrows():
        row_data = {}
        for col in page_df.columns:
            if col == "_explanations":
                row_data[col] = row[col] if isinstance(row[col], list) else []
            else:
                val = row[col]
                if isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = round(float(val), 4)
                elif isinstance(val, pd.Timestamp):
                    val = str(val)
                elif pd.isna(val):
                    val = None
                row_data[col] = val
        rows.append(row_data)

    return JSONResponse(content={
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
        "rows": rows,
    })


@app.get("/api/rules/{session_id}")
async def get_rules(session_id: str):
    """Get generated rules for a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    result = _sessions[session_id]["result"]
    rules_data = [r.to_dict() for r in result.rules]
    return JSONResponse(content={"rules": rules_data})


@app.put("/api/rules/{session_id}")
async def update_rules(session_id: str, request: Request):
    """Update rules (toggle, modify threshold) and re-evaluate."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    body = await request.json()
    updated_rules = body.get("rules", [])

    session = _sessions[session_id]
    engine = session["engine"]

    # Build Rule objects from sent data
    new_rules = [Rule.from_dict(r) for r in updated_rules]
    engine._last_result.rules = new_rules

    # Re-evaluate
    new_result = engine.rerun_evaluation()
    session["result"] = new_result

    response = _build_response(
        session_id, new_result,
        session["filename"], session["was_truncated"]
    )
    return JSONResponse(content=response)


@app.get("/api/profile/{session_id}")
async def get_profile(session_id: str):
    """Get data profile for a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    result = _sessions[session_id]["result"]
    return JSONResponse(content=result.profile.to_dict())


@app.get("/api/export/{session_id}")
async def export_results(session_id: str, kind: str = "flagged"):
    """Download results as CSV."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    result = _sessions[session_id]["result"]

    if kind == "flagged":
        df = result.result_df[result.result_df["_is_anomaly"] == 1].copy()
        fname = "flagged_anomalies.csv"
    elif kind == "rules":
        json_str = RuleGenerator.rules_to_json(result.rules)
        return StreamingResponse(
            io.BytesIO(json_str.encode()),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=rules.json"},
        )
    else:
        df = result.result_df.copy()
        fname = "full_results.csv"

    # Drop complex columns for CSV export
    if "_explanations" in df.columns:
        df["_explanations"] = df["_explanations"].apply(
            lambda x: json.dumps(x) if isinstance(x, list) else str(x)
        )

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={fname}"},
    )


@app.post("/api/ml/{session_id}")
async def run_ml(session_id: str, request: Request):
    """Run optional ML integrations."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    body = await request.json()
    enable_iforest = body.get("iforest", True)
    enable_xgboost = body.get("xgboost", False)

    session = _sessions[session_id]
    result = session["result"]

    enhanced_df = run_ml_integrations(
        result.result_df, result.profile,
        enable_iforest=enable_iforest,
        enable_xgboost=enable_xgboost,
    )
    result.result_df = enhanced_df
    session["result"] = result

    ml_cols = [c for c in enhanced_df.columns if c.startswith("_ml_")]
    ml_summary = {}
    for c in ml_cols:
        ml_summary[c] = {
            "mean_score": round(float(enhanced_df[c].mean()), 4),
            "flagged": int((enhanced_df[c] > 0.5).sum()),
        }

    return JSONResponse(content={"ml_results": ml_summary})


# %% [markdown]
# ## Helper Functions

# %%
def _build_response(session_id: str, result: EngineResult,
                    filename: str, was_truncated: bool) -> dict:
    """Build the JSON response for the frontend."""
    eval_r = result.evaluation
    profile = result.profile

    # Score distribution for chart
    scores = result.result_df["_anomaly_score"]
    hist, bin_edges = np.histogram(scores, bins=20, range=(0, 1))
    score_distribution = {
        "labels": [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(hist))],
        "values": hist.tolist(),
    }

    # Rule breakdown for chart
    rule_breakdown = []
    for rule in result.rules:
        count = eval_r.rule_trigger_counts.get(rule.rule_id, 0)
        rule_breakdown.append({
            "rule_id": rule.rule_id,
            "type": rule.rule_type,
            "description": rule.description,
            "severity": rule.severity,
            "triggered_count": count,
            "enabled": rule.enabled,
        })

    # Top anomalies preview (first 20)
    top_anomalies = []
    anomaly_df = result.result_df[result.result_df["_is_anomaly"] == 1].nlargest(20, "_anomaly_score")
    for _, row in anomaly_df.iterrows():
        row_preview = {}
        for col in result.result_df.columns:
            if col == "_explanations":
                row_preview[col] = row[col] if isinstance(row[col], list) else []
            elif col.startswith("_") and col not in ("_anomaly_score", "_is_anomaly", "_rules_triggered_count", "_explanations"):
                continue
            else:
                val = row[col]
                if isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = round(float(val), 4)
                elif isinstance(val, pd.Timestamp):
                    val = str(val)
                elif pd.isna(val):
                    val = None
                row_preview[col] = val
        top_anomalies.append(row_preview)

    # Column profile for display
    col_profiles = []
    for name, cp in profile.columns.items():
        col_profiles.append({
            "name": name,
            "dtype": cp.dtype,
            "semantic_role": cp.semantic_role,
            "null_rate": cp.stats.get("null_rate", 0),
            "unique_count": cp.stats.get("unique_count", 0),
        })

    return {
        "session_id": session_id,
        "filename": filename,
        "was_truncated": was_truncated,
        "summary": {
            "total_rows": eval_r.total_rows,
            "total_anomalies": eval_r.total_anomalies,
            "anomaly_rate": round(eval_r.anomaly_rate * 100, 2),
            "rules_evaluated": eval_r.rules_evaluated,
            "rules_skipped": eval_r.rules_skipped,
            "score_threshold": round(eval_r.score_threshold, 4),
        },
        "profile": {
            "detected_entity": profile.detected_entity_col,
            "detected_amount": profile.detected_amount_col,
            "detected_time": profile.detected_time_col,
            "detected_geo": list(profile.detected_geo_cols) if profile.detected_geo_cols else None,
            "detected_label": profile.detected_label_col,
            "detected_categories": profile.detected_category_cols,
            "columns": col_profiles,
        },
        "score_distribution": score_distribution,
        "rule_breakdown": rule_breakdown,
        "top_anomalies": top_anomalies,
        "rules": [r.to_dict() for r in result.rules],
    }


# %% [markdown]
# ## Entry Point

# %%
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Dynamic Rule Engine Dashboard...")
    print("   Open http://localhost:8000 in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
