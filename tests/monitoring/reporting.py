"""Reporting and baseline/diff utilities for quality metrics."""

import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import asdict
from loguru import logger

from .quality_dashboard import quality_dashboard, QualityMetrics


def export_json(filepath: str) -> None:
    data = [
        _metrics_to_dict(m) for m in quality_dashboard.metrics_history
    ]
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Exported JSON report to {filepath}")


def export_html(filepath: str) -> None:
    data = [
        _metrics_to_dict(m) for m in quality_dashboard.metrics_history
    ]
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    html = _render_html(data)
    with open(filepath, 'w') as f:
        f.write(html)
    logger.info(f"Exported HTML report to {filepath}")


def save_baseline(filepath: str) -> None:
    export_json(filepath)


def diff_against_baseline(baseline_path: str) -> Dict[str, Any]:
    try:
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
    except FileNotFoundError:
        return {"status": "missing_baseline"}

    current = [
        _metrics_to_dict(m) for m in quality_dashboard.metrics_history
    ]
    return _diff_metrics(baseline, current)


def _metrics_to_dict(m: QualityMetrics) -> Dict[str, Any]:
    return {
        "timestamp": m.timestamp.isoformat(),
        "response_id": m.response_id,
        "query": m.query,
        "response": m.response,
        "factual_accuracy": m.factual_accuracy,
        "completeness": m.completeness,
        "hallucination": m.hallucination,
        "utility": m.utility,
        "overall_score": m.overall_score,
    }


def _render_html(rows: List[Dict[str, Any]]) -> str:
    head = """
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Quality Report</title>
<style>
 body{font-family: Arial, sans-serif; padding: 16px;}
 table{border-collapse: collapse; width: 100%;}
 th,td{border:1px solid #ddd; padding:8px; font-size: 13px;}
 th{background:#f5f5f5; text-align:left;}
 .low{color:#b00020;} .high{color:#006400;}
</style></head><body>
<h2>Quality Report</h2>
<table>
<tr><th>Time</th><th>ID</th><th>Overall</th><th>Factual</th><th>Complete</th><th>Halluc.</th><th>Utility</th><th>Query</th></tr>
"""
    body_rows = []
    for r in rows:
        overall = r.get("overall_score", 0)
        factual = r.get("factual_accuracy", {}).get("overall_score", 0)
        complete = r.get("completeness", {}).get("overall_completeness", 0)
        halluc = 1.0 - r.get("hallucination", {}).get("overall_hallucination_score", 0)
        utility = r.get("utility", {}).get("overall_utility", 0)
        body_rows.append(
            f"<tr><td>{r['timestamp']}</td><td>{r['response_id']}</td>"
            f"<td>{overall:.3f}</td><td>{factual:.3f}</td><td>{complete:.3f}</td>"
            f"<td>{halluc:.3f}</td><td>{utility:.3f}</td><td>{r['query']}</td></tr>"
        )
    tail = """
</table>
</body></html>
"""
    return head + "\n".join(body_rows) + tail


def _diff_metrics(baseline: List[Dict[str, Any]], current: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Simple aggregate diff: compare means across key metrics
    def summarize(rows: List[Dict[str, Any]]):
        if not rows:
            return {"count": 0}
        def avg(vals):
            return sum(vals) / len(vals) if vals else 0.0
        overall = avg([r.get("overall_score", 0.0) for r in rows])
        factual = avg([r.get("factual_accuracy", {}).get("overall_score", 0.0) for r in rows])
        complete = avg([r.get("completeness", {}).get("overall_completeness", 0.0) for r in rows])
        halluc = avg([1.0 - r.get("hallucination", {}).get("overall_hallucination_score", 0.0) for r in rows])
        utility = avg([r.get("utility", {}).get("overall_utility", 0.0) for r in rows])
        return {
            "count": len(rows),
            "overall": overall,
            "factual": factual,
            "completeness": complete,
            "hallucination": halluc,
            "utility": utility,
        }

    base = summarize(baseline)
    curr = summarize(current)
    return {
        "baseline": base,
        "current": curr,
        "delta": {
            k: (curr.get(k, 0) - base.get(k, 0)) for k in ["overall", "factual", "completeness", "hallucination", "utility"]
        }
    }


