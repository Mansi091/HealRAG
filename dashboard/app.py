import os
import json
import logging
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

from health.health_checker import health_checker
from evaluation.baseline import baseline_manager
from evaluation.regression import regression_detector
from evidence.evidence_logger import evidence_logger

app = FastAPI(title="HealRAG Dashboard")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HealRAG Self-Healing Dashboard</title>
    <style>
        * { box-sizing: border-box; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { background-color: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 15px; margin-bottom: 25px; }
        .title { font-size: 24px; font-weight: bold; color: #38bdf8; display: flex; align-items: center; gap: 10px; }
        .badge { padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 14px; }
        .healthy { background-color: #166534; color: #4ade80; border: 1px solid #22c55e; }
        .degraded { background-color: #991b1b; color: #fca5a5; border: 1px solid #ef4444; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 25px; }
        .card { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; }
        .card h3 { margin-top: 0; color: #94a3b8; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
        .metric-value { font-size: 32px; font-weight: bold; color: #f8fafc; margin: 10px 0; }
        .subtext { font-size: 13px; color: #64748b; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { text-align: left; padding: 10px; border-bottom: 1px solid #334155; font-size: 14px; }
        th { color: #94a3b8; }
        .event-tag { font-size: 12px; padding: 2px 8px; border-radius: 4px; background: #334155; }
    </style>
</head>
<body>
    <div class="header">
        <div class="title">🛡️ HealRAG Self-Healing Dashboard</div>
        <div class="badge {STATUS_CLASS}">{HEALTH_STATUS}</div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>System Status</h3>
            <div class="metric-value">{HEALTH_STATUS}</div>
            <div class="subtext">{HEALTH_SUMMARY}</div>
        </div>
        <div class="card">
            <h3>Evaluation Precision</h3>
            <div class="metric-value">{PRECISION}%</div>
            <div class="subtext">Context Precision Target &ge; 70%</div>
        </div>
        <div class="card">
            <h3>Faithfulness</h3>
            <div class="metric-value">{FAITHFULNESS}%</div>
            <div class="subtext">Answer Grounding Target &ge; 70%</div>
        </div>
        <div class="card">
            <h3>Total Healing Events</h3>
            <div class="metric-value">{TOTAL_EVENTS}</div>
            <div class="subtext">Logged Self-Correction Interventions</div>
        </div>
    </div>

    <div class="card">
        <h3>Recent Evidence Journal Events</h3>
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Query</th>
                    <th>Failure Type</th>
                    <th>Action Taken</th>
                    <th>Attempt</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {TABLE_ROWS}
            </tbody>
        </table>
    </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    health = health_checker.check_health()
    baseline = baseline_manager.get_baseline() or {}
    events = evidence_logger.get_recent_events(limit=15)

    status = health.status
    status_class = "healthy" if status == "HEALTHY" else "degraded"

    precision = int(baseline.get("context_precision", 0.85) * 100)
    faithfulness = int(baseline.get("faithfulness", 0.90) * 100)

    rows = []
    if not events:
        rows.append("<tr><td colspan='6' style='text-align:center; color:#64748b;'>No evidence events recorded yet.</td></tr>")
    else:
        for e in reversed(events):
            rows.append(f"""
            <tr>
                <td>{e.get('timestamp', '')[:19]}</td>
                <td>{e.get('query', '')[:30]}...</td>
                <td><span class="event-tag">{e.get('failure_type', 'N/A')}</span></td>
                <td>{e.get('action', '')}</td>
                <td>#{e.get('attempt', 1)}</td>
                <td style="color:{'#4ade80' if e.get('status') in ['success', 'active'] else '#fca5a5'};">{e.get('status', 'success')}</td>
            </tr>
            """)

    html = HTML_TEMPLATE.replace("{HEALTH_STATUS}", status)\
                        .replace("{STATUS_CLASS}", status_class)\
                        .replace("{HEALTH_SUMMARY}", health.summary)\
                        .replace("{PRECISION}", str(precision))\
                        .replace("{FAITHFULNESS}", str(faithfulness))\
                        .replace("{TOTAL_EVENTS}", str(len(events)))\
                        .replace("{TABLE_ROWS}", "".join(rows))

    return HTMLResponse(content=html)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)
