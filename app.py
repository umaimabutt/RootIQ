
import os
import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Optional provider SDKs. The app gives a friendly message if one is missing.
try:
    from groq import Groq
except Exception:
    Groq = None



# ============================================================
# RootIQ configuration
# ============================================================

APP_NAME = "RootIQ"
APP_SUBTITLE = 'AI that finds the "why" behind your business data.'
GROQ_MODEL = "openai/gpt-oss-120b"

COLUMN_SYNONYMS = {
    "date": ["date", "order_date", "transaction_date", "purchase_date", "month",
             "week", "day", "period", "timestamp", "created_at"],
    "revenue": ["revenue", "sales", "total_sales", "gross_revenue", "net_sales",
                "income", "amount", "total_amount"],
    "cost": ["cost", "expense", "expenses", "cogs", "expenditure", "spend", "total_cost"],
    "profit": ["profit", "net_profit", "margin", "net_income", "gross_profit"],
    "orders": ["orders", "order_count", "quantity", "qty", "units", "units_sold",
               "transactions", "num_orders"],
    "customer": ["customer", "customer_id", "client", "client_id", "buyer", "user_id"],
    "product": ["product", "product_name", "sku", "item", "item_name"],
    "segment": ["category", "product_category", "segment", "region", "location",
                "channel", "country", "state", "city", "market"],
    "marketing_spend": ["ad_spend", "marketing_spend", "advertising", "ad_cost",
                        "campaign_spend", "marketing_cost"],
    "conversion": ["conversion_rate", "cvr", "conversion"],
    "traffic": ["visits", "website_visits", "sessions", "traffic", "page_views", "impressions"],
}

PRIMARY_METRICS = ["revenue", "profit", "orders", "cost", "marketing_spend", "traffic", "conversion"]
SEVERITY_HIGH_CHANGE = 0.25
SEVERITY_MEDIUM_CHANGE = 0.10
ANOMALY_CHANGE_THRESHOLD = 0.20
REQUIRED_AI_KEYS = [
    "executive_summary", "key_findings", "detected_problems",
    "root_causes", "recommendations", "additional_data_needed", "next_actions"
]

SYSTEM_PROMPT = """
You are RootIQ, an evidence-based business analytics and root-cause analysis assistant.

CRITICAL RULES:
1. You receive structured findings computed by Python/Pandas from the user's uploaded data.
2. The uploaded data and computed findings are your ONLY source of business facts.
3. Never invent numbers, dates, columns, customers, products, events, causes, or evidence.
4. Correlation does NOT prove causation. Use wording such as "associated with", "may indicate",
   "likely", or "could be consistent with". Do not claim that X caused Y from correlation alone.
5. Every problem, root-cause hypothesis, and recommendation must trace to evidence and source columns
   contained in the payload.
6. If evidence is insufficient, say exactly:
   "Insufficient evidence to determine the root cause."
   Then explain what additional data would improve confidence.
7. Do not give generic recommendations such as "improve marketing" or "increase sales".
   Recommendations must identify a mechanism and connect to the observed evidence.
8. Do not use outside web knowledge. If the user asks something not answerable from the uploaded
   material, say that RootIQ can only answer from the uploaded data.
9. Treat numeric values in the payload as ground truth.
10. Return JSON only. No Markdown fences and no prose outside the JSON.

Return exactly this top-level structure:
{
  "executive_summary": "string",
  "key_findings": ["string"],
  "detected_problems": [
    {
      "problem": "string",
      "severity": "High|Medium|Low",
      "evidence": "string",
      "affected_metric": "string",
      "time_period": "string",
      "magnitude": 0.0
    }
  ],
  "root_causes": [
    {
      "problem": "string",
      "hypothesis": "string",
      "supporting_evidence": ["string"],
      "contradicting_evidence": ["string"],
      "confidence": "High|Medium|Low",
      "evidence_sources": ["string"]
    }
  ],
  "recommendations": [
    {
      "problem": "string",
      "recommendation": "string",
      "priority": "High|Medium|Low",
      "expected_impact": "High|Medium|Low",
      "reason": "string"
    }
  ],
  "additional_data_needed": ["string"],
  "next_actions": ["string"]
}
"""


class DataLoadError(Exception):
    pass


class AIProviderError(Exception):
    pass


# ============================================================
# Data loading and profiling
# ============================================================

def load_data(uploaded_file) -> Tuple[pd.DataFrame, str]:
    if uploaded_file is None:
        raise DataLoadError("Please upload a CSV, Excel, JSON, or PDF file.")

    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if not raw:
        raise DataLoadError("The uploaded file is empty.")

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw))
            file_type = "CSV"
        elif name.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
            file_type = "Excel"
        elif name.endswith(".json"):
            try:
                df = pd.read_json(io.BytesIO(raw))
            except ValueError:
                obj = json.loads(raw.decode("utf-8"))
                if isinstance(obj, list):
                    df = pd.DataFrame(obj)
                elif isinstance(obj, dict):
                    df = pd.DataFrame(obj)
                else:
                    raise DataLoadError("The JSON structure could not be converted to a table.")
            file_type = "JSON"
        elif name.endswith(".pdf"):
            try:
                import pdfplumber
            except Exception:
                raise DataLoadError("PDF support is not installed. Redeploy the app so the updated requirements.txt is installed.")
            tables = []
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                for page in pdf.pages:
                    for table in (page.extract_tables() or []):
                        if not table or len(table) < 2:
                            continue
                        rows = [[cell.strip() if isinstance(cell, str) else cell for cell in row] for row in table]
                        header = rows[0]
                        if not any(str(x or "").strip() for x in header):
                            continue
                        columns = [str(x or "Unnamed").strip() for x in header]
                        data_rows = rows[1:]
                        table_df = pd.DataFrame(data_rows, columns=columns).dropna(how="all")
                        if not table_df.empty:
                            tables.append(table_df)
            if not tables:
                raise DataLoadError("No readable table was found in this PDF. RootIQ supports PDFs with selectable tables. Scanned/image PDFs need a CSV or Excel version.")
            df = pd.concat(tables, ignore_index=True, sort=False)
            file_type = "PDF"
        else:
            raise DataLoadError("Unsupported file type. Use CSV, XLSX, JSON, or PDF.")
    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(f"Could not read the file. Please check that it is valid. ({exc})")

    if df is None or df.empty:
        raise DataLoadError("The uploaded file contains no usable rows.")
    if len(df.columns) == 0:
        raise DataLoadError("The uploaded file contains no columns.")

    df = df.copy()
    df.columns = make_unique_columns([str(c).strip() for c in df.columns])
    return df, file_type

def make_unique_columns(columns: List[str]) -> List[str]:
    seen = {}
    out = []
    for col in columns:
        base = col or "Unnamed"
        count = seen.get(base, 0)
        out.append(base if count == 0 else f"{base}_{count}")
        seen[base] = count + 1
    return out


def normalize_name(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    datetime_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)

    missing = {str(c): int(df[c].isna().sum()) for c in df.columns}
    unique = {str(c): int(df[c].nunique(dropna=True)) for c in df.columns}

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "column_names": [str(c) for c in df.columns],
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "missing_values": missing,
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_values": unique,
        "numeric_columns": [str(c) for c in numeric],
        "categorical_columns": [str(c) for c in categorical],
        "datetime_columns": [str(c) for c in datetime_cols],
    }


def detect_columns(df: pd.DataFrame, date_override: str = "", target_override: str = "") -> Dict[str, Dict[str, Any]]:
    normalized = {col: normalize_name(col) for col in df.columns}
    result = {}

    def score_match(col_norm: str, synonym: str) -> int:
        if col_norm == synonym:
            return 100
        if col_norm.replace("_", "") == synonym.replace("_", ""):
            return 95
        if synonym in col_norm:
            return 70
        return 0

    for role, synonyms in COLUMN_SYNONYMS.items():
        candidates = []
        for col, col_norm in normalized.items():
            best = max((score_match(col_norm, normalize_name(s)) for s in synonyms), default=0)
            if best > 0:
                dtype_bonus = 10 if role in {"revenue", "cost", "profit", "orders",
                                             "marketing_spend", "conversion", "traffic"} and pd.api.types.is_numeric_dtype(df[col]) else 0
                if role == "date":
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    dtype_bonus = 15 if parsed.notna().mean() > 0.90 else 0
                candidates.append((best + dtype_bonus, col))

        if candidates:
            candidates.sort(reverse=True)
            score, col = candidates[0]
            confidence = "High" if score >= 100 else "Medium" if score >= 70 else "Low"
            result[role] = {"column": col, "confidence": confidence, "score": int(score)}
        else:
            result[role] = {"column": None, "confidence": "None", "score": 0}

    # Date dtype fallback.
    if not result["date"]["column"]:
        best_date = None
        best_ratio = 0
        for col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            ratio = float(parsed.notna().mean())
            if ratio > best_ratio and ratio > 0.90:
                best_ratio, best_date = ratio, col
        if best_date:
            result["date"] = {"column": best_date, "confidence": "Low", "score": 50}

    if date_override and date_override in df.columns:
        result["date"] = {"column": date_override, "confidence": "User override", "score": 999}

    if target_override and target_override in df.columns:
        result["target_metric"] = {"column": target_override, "confidence": "User override", "score": 999}
    else:
        target = next((result[r]["column"] for r in ["revenue", "profit", "orders"] if result[r]["column"]), None)
        result["target_metric"] = {"column": target, "confidence": "Auto", "score": 0}

    return result


# ============================================================
# Analytics
# ============================================================

def safe_float(value):
    if value is None or pd.isna(value) or not np.isfinite(float(value)):
        return None
    return float(value)


def pct_change(first, last):
    if first is None or last is None or pd.isna(first) or pd.isna(last) or float(first) == 0:
        return None
    return float((last - first) / abs(first))


def compute_descriptive_stats(df: pd.DataFrame, column_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    stats = {}
    cols = []
    for role in PRIMARY_METRICS:
        col = column_map.get(role, {}).get("column")
        if col and col not in cols and pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)

    if not cols:
        return {}

    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        stats[col] = {
            "mean": safe_float(s.mean()),
            "median": safe_float(s.median()),
            "min": safe_float(s.min()),
            "max": safe_float(s.max()),
            "std": safe_float(s.std()),
            "pct_change": pct_change(s.iloc[0], s.iloc[-1]),
        }
    return stats


def prepare_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    work = df.copy()
    work["_ri_date"] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=["_ri_date"]).sort_values("_ri_date")
    return work


def choose_frequency(date_series: pd.Series) -> str:
    if len(date_series) < 60:
        return "D"
    span_days = max((date_series.max() - date_series.min()).days, 1)
    points_per_day = len(date_series) / span_days
    if span_days > 365 or points_per_day < 0.5:
        return "ME"
    return "W"


def compute_time_series_analysis(df: pd.DataFrame, column_map: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    date_col = column_map.get("date", {}).get("column")
    if not date_col:
        return None

    work = prepare_datetime(df, date_col)
    if len(work) < 3:
        return None

    freq = choose_frequency(work["_ri_date"])
    metrics = []
    for role in ["revenue", "profit", "orders", "cost", "marketing_spend", "traffic", "conversion"]:
        col = column_map.get(role, {}).get("column")
        if col and col not in metrics and pd.api.types.is_numeric_dtype(work[col]):
            metrics.append(col)

    result = {"date_column": date_col, "frequency": freq, "metrics": {}, "trend_table": {}}

    for col in metrics:
        ts = work.set_index("_ri_date")[col].resample(freq).sum(min_count=1).dropna()
        if len(ts) < 2:
            continue
        growth = ts.pct_change().replace([np.inf, -np.inf], np.nan)
        largest_idx = growth.abs().idxmax()
        direction = "upward" if ts.iloc[-1] > ts.iloc[0] else "downward" if ts.iloc[-1] < ts.iloc[0] else "flat"
        result["metrics"][col] = {
            "trend_direction": direction,
            "first_value": safe_float(ts.iloc[0]),
            "last_value": safe_float(ts.iloc[-1]),
            "overall_change": pct_change(ts.iloc[0], ts.iloc[-1]),
            "largest_period_change": safe_float(growth.loc[largest_idx]),
            "largest_period_date": str(largest_idx.date()) if hasattr(largest_idx, "date") else str(largest_idx),
            "points": int(len(ts)),
        }
        result["trend_table"][col] = ts.reset_index().rename(columns={col: "value"}).to_dict(orient="records")
    return result


def detect_anomalies(df: pd.DataFrame, column_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    anomalies = []
    date_col = column_map.get("date", {}).get("column")
    metrics = []
    for role in PRIMARY_METRICS:
        col = column_map.get(role, {}).get("column")
        if col and col not in metrics and pd.api.types.is_numeric_dtype(df[col]):
            metrics.append(col)

    work = prepare_datetime(df, date_col) if date_col else df.copy()

    for col in metrics:
        s = pd.to_numeric(work[col], errors="coerce").dropna()
        if len(s) < 5:
            continue

        skew = float(s.skew()) if len(s) >= 3 else 0.0
        method = "IQR"
        flagged = pd.Series(False, index=s.index)
        expected_low, expected_high = None, None

        if date_col and len(s) >= 12:
            # Local-trend anomaly detection.
            roll = s.rolling(window=min(7, max(3, len(s) // 3)), min_periods=3)
            baseline = roll.mean()
            local_std = roll.std().replace(0, np.nan)
            z = (s - baseline) / local_std
            flagged = z.abs() > 2.5
            method = "rolling z-score"
            expected_low = (baseline - 2.5 * local_std)
            expected_high = (baseline + 2.5 * local_std)
        elif len(s) >= 30 and abs(skew) < 1:
            z = (s - s.mean()) / (s.std() if s.std() else 1)
            flagged = z.abs() > 2.5
            method = "z-score"
            expected_low = pd.Series(s.mean() - 2.5 * s.std(), index=s.index)
            expected_high = pd.Series(s.mean() + 2.5 * s.std(), index=s.index)
        else:
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            flagged = (s < low) | (s > high)
            method = "IQR"
            expected_low = pd.Series(low, index=s.index)
            expected_high = pd.Series(high, index=s.index)

        # Always surface a large period-over-period movement.
        if len(s) >= 2:
            change = s.pct_change().replace([np.inf, -np.inf], np.nan)
            flagged = flagged | (change.abs() > ANOMALY_CHANGE_THRESHOLD)

        flagged_indices = list(s.index[flagged])[:15]
        for idx in flagged_indices:
            value = s.loc[idx]
            date_value = work.loc[idx, "_ri_date"] if "_ri_date" in work.columns and idx in work.index else idx
            anomalies.append({
                "column": col,
                "date": str(date_value.date()) if hasattr(date_value, "date") else str(date_value),
                "value": safe_float(value),
                "expected_range": [
                    safe_float(expected_low.loc[idx]) if expected_low is not None and idx in expected_low.index else None,
                    safe_float(expected_high.loc[idx]) if expected_high is not None and idx in expected_high.index else None,
                ],
                "method": method,
            })

    return anomalies[:50]


def compute_correlations(df: pd.DataFrame, column_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    cols = []
    for role in PRIMARY_METRICS:
        col = column_map.get(role, {}).get("column")
        if col and col not in cols and pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)

    if len(cols) < 2:
        return {"pairs": [], "matrix": {}}

    corr = df[cols].corr(numeric_only=True)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) > 0.3:
                pairs.append({"pair": [a, b], "r": safe_float(r), "type": "correlation"})
    pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
    return {"pairs": pairs[:5], "matrix": corr.round(3).to_dict()}


def compute_segmentation(df: pd.DataFrame, column_map: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    segment_col = column_map.get("segment", {}).get("column") or column_map.get("product", {}).get("column")
    metric = (column_map.get("revenue", {}).get("column")
              or column_map.get("profit", {}).get("column")
              or column_map.get("orders", {}).get("column"))
    if not segment_col or not metric or not pd.api.types.is_numeric_dtype(df[metric]):
        return None

    grouped = df.groupby(segment_col, dropna=False)[metric].agg(["sum", "mean", "count"]).reset_index()
    if grouped.empty:
        return None
    grouped = grouped.sort_values("sum", ascending=False)
    values = grouped["sum"]
    std = values.std()
    mean = values.mean()
    grouped["outlier"] = (grouped["sum"] - mean).abs() > (1.5 * std if std else np.inf)

    return {
        "segment_column": segment_col,
        "metric": metric,
        "top_performers": grouped.head(5).to_dict(orient="records"),
        "bottom_performers": grouped.tail(5).sort_values("sum").to_dict(orient="records"),
        "outliers": grouped[grouped["outlier"]].head(10).to_dict(orient="records"),
    }


def detect_problems(stats, ts, anomalies, correlations, segmentation, column_map) -> List[Dict[str, Any]]:
    problems = []
    candidates = []

    for role in ["revenue", "profit", "orders", "cost", "marketing_spend"]:
        col = column_map.get(role, {}).get("column")
        if not col:
            continue

        info = ts.get("metrics", {}).get(col) if ts else None
        change = info.get("overall_change") if info else stats.get(col, {}).get("pct_change")
        if change is None:
            continue

        if role in {"revenue", "profit", "orders"} and change < -SEVERITY_MEDIUM_CHANGE:
            name = f"{role.replace('_', ' ').title()} decline"
        elif role == "cost" and change > SEVERITY_MEDIUM_CHANGE:
            name = "Rising costs"
        elif role == "marketing_spend" and change > SEVERITY_MEDIUM_CHANGE:
            name = "Rising marketing spend"
        else:
            continue

        severity = "High" if abs(change) >= SEVERITY_HIGH_CHANGE else "Medium"
        if ts and info and abs(info.get("largest_period_change") or 0) >= SEVERITY_HIGH_CHANGE:
            severity = "High"

        period = ""
        if info:
            period = f"largest change around {info.get('largest_period_date', 'the detected period')}"
        candidates.append({
            "problem": name,
            "severity": severity,
            "metric": col,
            "evidence": f"{col} changed {change:.1%} from the first to the last available period.",
            "time_period": period or "available data range",
            "magnitude": float(change),
            "source_columns": [col] + ([column_map["date"]["column"]] if column_map.get("date", {}).get("column") else []),
        })

    if anomalies:
        by_col = {}
        for a in anomalies:
            by_col.setdefault(a["column"], 0)
            by_col[a["column"]] += 1
        for col, count in sorted(by_col.items(), key=lambda x: x[1], reverse=True)[:3]:
            candidates.append({
                "problem": f"Unusual movement in {col}",
                "severity": "Medium",
                "metric": col,
                "evidence": f"{count} unusual observation(s) were flagged by the selected anomaly method.",
                "time_period": "flagged observations",
                "magnitude": 0.0,
                "source_columns": [col] + ([column_map["date"]["column"]] if column_map.get("date", {}).get("column") else []),
            })

    if segmentation:
        outliers = segmentation.get("outliers", [])
        if outliers:
            candidates.append({
                "problem": f"Segment imbalance in {segmentation['segment_column']}",
                "severity": "Medium",
                "metric": segmentation["metric"],
                "evidence": f"Some {segmentation['segment_column']} groups are more than ~1.5 standard deviations from the group mean.",
                "time_period": "available data",
                "magnitude": 0.0,
                "source_columns": [segmentation["segment_column"], segmentation["metric"]],
            })

    # De-duplicate by problem + metric.
    seen = set()
    for item in sorted(candidates, key=lambda x: ({"High": 0, "Medium": 1, "Low": 2}[x["severity"]], -abs(x["magnitude"]))):
        key = (item["problem"], item["metric"])
        if key not in seen:
            problems.append(item)
            seen.add(key)
    return problems[:10]


def build_root_cause_hypotheses(problems, correlations, anomalies, segmentation, column_map):
    hypotheses = []
    corr_pairs = correlations.get("pairs", []) if correlations else []

    for problem in problems:
        metric = problem["metric"]
        supporting = []
        sources = list(problem["source_columns"])

        for pair in corr_pairs:
            if metric in pair["pair"]:
                other = pair["pair"][0] if pair["pair"][1] == metric else pair["pair"][1]
                supporting.append(f"{metric} and {other} moved together (correlation r={pair['r']:.2f}); this is an association, not proof of causation.")
                sources.extend(pair["pair"])

        related_anomalies = [a for a in anomalies if a["column"] == metric][:3]
        if related_anomalies:
            supporting.append(f"{metric} has {len([a for a in anomalies if a['column'] == metric])} flagged unusual observation(s).")
            sources.append(metric)

        if segmentation and metric == segmentation.get("metric"):
            supporting.append(f"Performance differs materially across {segmentation['segment_column']} groups.")
            sources.append(segmentation["segment_column"])

        if supporting:
            confidence = "Medium"
            if any(abs(p["r"]) > 0.6 for p in corr_pairs if metric in p["pair"]):
                confidence = "High"
            hypotheses.append({
                "problem": problem["problem"],
                "hypothesis": f"{problem['problem']} may be associated with changes in related metrics or segments shown in the evidence.",
                "supporting_evidence": supporting,
                "contradicting_evidence": [],
                "confidence": confidence,
                "evidence_sources": sorted(set(sources)),
            })
        else:
            hypotheses.append({
                "problem": problem["problem"],
                "hypothesis": "Insufficient evidence to determine the root cause.",
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "confidence": "Low",
                "evidence_sources": problem["source_columns"],
            })
    return hypotheses


def build_recommendations(problems, hypotheses):
    recs = []
    for p, h in zip(problems, hypotheses):
        if h["hypothesis"].startswith("Insufficient evidence"):
            text = f"Collect additional data related to {p['metric']} and the period identified before taking a major corrective action."
            reason = "The current uploaded data does not provide enough evidence for a specific root cause."
            priority = p["severity"]
        else:
            text = f"Investigate the metrics and segments linked to {p['metric']} in the evidence before changing strategy."
            reason = "The recommendation is tied to the observed evidence and avoids treating correlation as causation."
            priority = p["severity"]
        recs.append({
            "problem": p["problem"],
            "recommendation": text,
            "priority": priority,
            "expected_impact": priority,
            "reason": reason,
        })
    return recs


# ============================================================
# AI payload, providers, defensive JSON parsing
# ============================================================

def build_ai_payload(df, business_type, dataset_description, profile, column_map, stats, ts, anomalies, correlations, segmentation, problems, hypotheses):
    sample = df.head(8).copy()
    sample = sample.replace({np.nan: None})
    sample_rows = sample.to_dict(orient="records")

    date_col = column_map.get("date", {}).get("column")
    date_range = None
    if date_col:
        parsed = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if not parsed.empty:
            date_range = f"{parsed.min().date()} – {parsed.max().date()}"

    payload = {
        "business_context": {
            "business_type": business_type or "Not provided",
            "dataset_description": dataset_description or "Not provided",
        },
        "profile_summary": {
            "row_count": profile["row_count"],
            "column_count": profile["column_count"],
            "date_range": date_range,
            "detected_columns": {k: v.get("column") for k, v in column_map.items()},
            "missing_values": profile["missing_values"],
            "duplicate_rows": profile["duplicate_rows"],
        },
        "descriptive_stats": stats,
        "time_series": ts.get("metrics", {}) if ts else {},
        "anomalies": anomalies[:20],
        "correlations": correlations.get("pairs", []) if correlations else [],
        "segmentation": segmentation or {},
        "detected_problems": problems[:10],
        "root_cause_scaffolding": hypotheses[:10],
        "sample_rows": sample_rows,
    }
    return payload


def get_secret(name: str) -> Optional[str]:
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get(name)


def groq_call(system_prompt: str, payload_json: str) -> str:
    if Groq is None:
        raise AIProviderError("Groq SDK is not installed. Run pip install -r requirements.txt.")
    key = get_secret("GROQ_API_KEY")
    if not key:
        raise AIProviderError("GROQ_API_KEY is missing. Add it to Streamlit Secrets or your environment.")
    try:
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload_json},
            ],
            temperature=0.2,
            max_completion_tokens=6000,
        )
        return response.choices[0].message.content
    except Exception as exc:
        raise AIProviderError(f"Groq request failed: {exc}")


def analyze_with_ai(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    payload_json = json.dumps(payload, ensure_ascii=False, default=str)
    raw = groq_call(SYSTEM_PROMPT, payload_json)
    parsed, raw_fallback = parse_ai_json(raw)
    return parsed, raw_fallback, raw


# ============================================================
# Dataset-only chatbot
# ============================================================

def build_chat_payload(question: str, payload: Dict[str, Any], chat_history: List[Dict[str, str]]) -> str:
    chat_prompt = """
You are the RootIQ document-only analyst.

Answer the user's question using ONLY the uploaded dataset and the computed RootIQ findings supplied below.
Do not use outside facts, web knowledge, assumptions, or invented values.
If the question cannot be answered from the uploaded material, say:
"I can't answer that from the uploaded data."

For numeric questions, use the supplied computed findings when available and do not invent calculations.
For causal questions, distinguish association from causation.
Keep answers clear and concise.

USER QUESTION:
""" + question + """

ROOTIQ FINDINGS:
""" + json.dumps(payload, ensure_ascii=False, default=str)[:45000]
    return chat_prompt


def chat_with_dataset(question, payload, history):
    prompt = build_chat_payload(question, payload, history)
    return groq_chat_text(prompt)

def groq_chat_text(prompt):
    if Groq is None:
        raise AIProviderError("Groq SDK is not installed.")
    key = get_secret("GROQ_API_KEY")
    if not key:
        raise AIProviderError("GROQ_API_KEY is missing.")
    client = Groq(api_key=key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=2500,
    )
    return response.choices[0].message.content



# ============================================================
# Markdown report
# ============================================================

def generate_markdown_report(results: Dict[str, Any]) -> str:
    ai = results.get("ai") or {}
    lines = [
        "# RootIQ Business Analysis Report",
        "",
        f"## Executive Summary\n{ai.get('executive_summary', 'AI summary unavailable.')}",
        "",
        "## Key Findings",
    ]
    for x in ai.get("key_findings", []):
        lines.append(f"- {x}")

    lines += ["", "## Detected Problems"]
    for p in ai.get("detected_problems", []):
        lines.append(f"### {p.get('problem', 'Problem')} — {p.get('severity', 'Unknown')}")
        lines.append(f"- Evidence: {p.get('evidence', '')}")
        lines.append(f"- Metric: {p.get('affected_metric', '')}")
        lines.append(f"- Period: {p.get('time_period', '')}")
        lines.append(f"- Magnitude: {p.get('magnitude', '')}")

    lines += ["", "## Root-Cause Analysis"]
    for r in ai.get("root_causes", []):
        lines.append(f"### {r.get('problem', '')}")
        lines.append(f"**Hypothesis:** {r.get('hypothesis', '')}")
        lines.append(f"**Confidence:** {r.get('confidence', '')}")
        lines.append("**Supporting evidence:**")
        lines.extend([f"- {x}" for x in r.get("supporting_evidence", [])])
        lines.append("**Contradicting evidence:**")
        lines.extend([f"- {x}" for x in r.get("contradicting_evidence", [])])
        lines.append(f"**Evidence sources:** {', '.join(r.get('evidence_sources', []))}")

    lines += ["", "## Recommendations"]
    for r in ai.get("recommendations", []):
        lines.append(f"### {r.get('recommendation', '')}")
        lines.append(f"- Priority: {r.get('priority', '')}")
        lines.append(f"- Expected impact: {r.get('expected_impact', '')}")
        lines.append(f"- Reason: {r.get('reason', '')}")

    lines += ["", "## Additional Data Needed"]
    lines.extend([f"- {x}" for x in ai.get("additional_data_needed", [])])
    lines += ["", "## Next Actions"]
    lines.extend([f"- {x}" for x in ai.get("next_actions", [])])
    return "\n".join(lines)


# ============================================================
# UI
# ============================================================

def badge(text: str, kind: str = "neutral"):
    cls = {"High": "high", "Medium": "medium", "Low": "low"}.get(text, kind)
    return f'<span class="badge {cls}">{text}</span>'


def render_dashboard(results):
    profile = results["profile"]
    column_map = results["column_map"]
    stats = results["stats"]
    ts = results["ts"]
    corr = results["correlations"]
    seg = results["segmentation"]
    ai = results.get("ai") or {}

    st.markdown("## Executive Summary")
    st.info(ai.get("executive_summary", "AI summary unavailable. See the Python-generated findings below."))

    kpi_cols = []
    for role, label in [("revenue", "Revenue"), ("profit", "Profit"), ("orders", "Orders")]:
        col = column_map.get(role, {}).get("column")
        if col and col in stats:
            value = stats[col].get("mean")
            latest = stats[col].get("max")
            kpi_cols.append((label, latest, stats[col].get("pct_change")))
    if kpi_cols:
        cards = st.columns(len(kpi_cols))
        for c, (label, value, change) in zip(cards, kpi_cols):
            c.metric(label, f"{value:,.2f}" if value is not None else "—",
                     f"{change:.1%}" if change is not None else None)

    st.divider()
    st.subheader("Data Overview")
    a, b, c, d = st.columns(4)
    a.metric("Rows", profile["row_count"])
    b.metric("Columns", profile["column_count"])
    c.metric("Duplicates", profile["duplicate_rows"])
    date_col = column_map.get("date", {}).get("column")
    if date_col:
        parsed = pd.to_datetime(results["df"][date_col], errors="coerce").dropna()
        d.metric("Date range", f"{parsed.min().date()} → {parsed.max().date()}" if not parsed.empty else "—")
    else:
        d.metric("Date analysis", "Not available")

    st.caption("Auto-detected roles: " + ", ".join(
        f"{r}={v['column']} ({v['confidence']})" for r, v in column_map.items()
        if v.get("column")
    ))

    if ts and ts.get("trend_table"):
        st.divider()
        st.subheader("Trends")
        for col, records in list(ts["trend_table"].items())[:4]:
            chart_df = pd.DataFrame(records)
            if not chart_df.empty:
                fig = px.line(chart_df, x="_ri_date", y="value", title=col)
                st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Detected Problems")
    problems = ai.get("detected_problems") or results["problems"]
    if not problems:
        st.success("No major evidence-backed problem was detected by the current rules.")
    for p in problems:
        sev = p.get("severity", "Low")
        with st.container(border=True):
            st.markdown(f"### {p.get('problem', 'Problem')} {badge(sev)}", unsafe_allow_html=True)
            st.write(p.get("evidence", ""))
            st.caption(
                f"Metric: {p.get('affected_metric', p.get('metric', ''))}  •  "
                f"Period: {p.get('time_period', '')}  •  "
                f"Source columns: {', '.join(p.get('source_columns', [])) if p.get('source_columns') else 'see evidence'}"
            )

    st.divider()
    st.subheader("Root-Cause Analysis")
    causes = ai.get("root_causes") or results["hypotheses"]
    for r in causes:
        with st.container(border=True):
            st.markdown(f"### {r.get('problem', '')} {badge(r.get('confidence', 'Low'))}", unsafe_allow_html=True)
            st.write(r.get("hypothesis", ""))
            left, right = st.columns(2)
            with left:
                st.markdown("**Supporting evidence**")
                for x in r.get("supporting_evidence", []):
                    st.write("✓ " + x)
            with right:
                st.markdown("**Contradicting evidence**")
                if r.get("contradicting_evidence"):
                    for x in r["contradicting_evidence"]:
                        st.write("• " + x)
                else:
                    st.write("None identified in the supplied findings.")
            st.caption("Evidence sources: " + ", ".join(r.get("evidence_sources", [])))

    st.divider()
    st.subheader("Recommendations")
    recs = ai.get("recommendations") or results["recommendations"]
    for r in recs:
        with st.container(border=True):
            st.markdown(f"**{r.get('recommendation', '')}**")
            st.write(r.get("reason", ""))
            st.caption(
                f"Priority: {r.get('priority', 'Low')} • "
                f"Expected impact: {r.get('expected_impact', 'Low')}"
            )

    st.divider()
    st.subheader("Evidence Explorer")
    date_col = column_map.get("date", {}).get("column")
    if date_col:
        for r in causes[:5]:
            with st.expander(f"Evidence for: {r.get('problem', '')}"):
                sources = r.get("evidence_sources", [])
                existing = [x for x in sources if x in results["df"].columns]
                if existing:
                    st.dataframe(results["df"][existing].head(50), use_container_width=True)
                else:
                    st.caption("No directly filterable source columns were identified.")

    report = generate_markdown_report(results)
    st.download_button(
        "⬇️ Download Markdown Report",
        data=report,
        file_name="rootiq_report.md",
        mime="text/markdown",
        use_container_width=True,
    )


def main():
    st.set_page_config(page_title="RootIQ", page_icon="🔎", layout="wide")

    st.markdown("""
    <style>
    .block-container {max-width: 1200px; padding-top: 2rem;}
    .hero {padding: 1.2rem 1.4rem; border-radius: 18px; border: 1px solid rgba(128,128,128,.25);
           background: linear-gradient(135deg, rgba(80,90,255,.12), rgba(0,180,160,.08)); margin-bottom: 1rem;}
    .hero h1 {margin:0; font-size: 2.4rem;}
    .hero p {margin:.35rem 0 0; opacity:.75;}
    .badge {padding: .15rem .5rem; border-radius: 999px; font-size: .75rem; font-weight: 700;}
    .high {background:#ffdddd; color:#a00000;}
    .medium {background:#fff1cc; color:#875500;}
    .low {background:#ddf6e7; color:#176b3a;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="hero">
      <h1>🔎 {APP_NAME}</h1>
      <p>{APP_SUBTITLE}</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("RootIQ Setup")
        uploaded = st.file_uploader("Upload business data", type=["csv", "xlsx", "json", "pdf"])
        business_type = st.text_input("Business type", placeholder="e.g. Online clothing store")
        dataset_description = st.text_area(
            "Dataset description",
            placeholder="e.g. Monthly sales, orders, ad spend, website visits..."
        )

        with st.expander("Advanced"):
            industry = st.text_input("Industry")
            goal = st.text_input("Main business goal")
            key_kpi = st.text_input("Key KPI")
            date_override = st.text_input("Date column override")
            target_override = st.text_input("Target metric override")

        analyze = st.button("🚀 Analyze with RootIQ", type="primary", use_container_width=True)

    if analyze:
        st.session_state.pop("rootiq_results", None)
        if uploaded is None:
            st.error("Upload a CSV, XLSX, JSON, or PDF file first.")
            return

        with st.spinner("Loading, profiling, and analyzing your data..."):
            try:
                df, file_type = load_data(uploaded)
                profile = profile_dataset(df)
                column_map = detect_columns(df, date_override, target_override)
                stats = compute_descriptive_stats(df, column_map)
                ts = compute_time_series_analysis(df, column_map)
                anomalies = detect_anomalies(df, column_map)
                correlations = compute_correlations(df, column_map)
                segmentation = compute_segmentation(df, column_map)
                problems = detect_problems(stats, ts, anomalies, correlations, segmentation, column_map)
                hypotheses = build_root_cause_hypotheses(problems, correlations, anomalies, segmentation, column_map)
                recommendations = build_recommendations(problems, hypotheses)
                payload = build_ai_payload(
                    df, business_type, dataset_description, profile, column_map,
                    stats, ts, anomalies, correlations, segmentation, problems, hypotheses
                )

                ai = None
                raw_ai = None
                ai_error = None
                try:
                    ai, raw_ai, _ = analyze_with_ai(payload)
                except AIProviderError as exc:
                    ai_error = str(exc)

                st.session_state["rootiq_results"] = {
                    "df": df,
                    "file_type": file_type,
                    "profile": profile,
                    "column_map": column_map,
                    "stats": stats,
                    "ts": ts,
                    "anomalies": anomalies,
                    "correlations": correlations,
                    "segmentation": segmentation,
                    "problems": problems,
                    "hypotheses": hypotheses,
                    "recommendations": recommendations,
                    "payload": payload,
                    "ai": ai,
                    "raw_ai": raw_ai,
                    "ai_error": ai_error,
                    "business_type": business_type,
                    "dataset_description": dataset_description,
                    "industry": industry,
                    "goal": goal,
                    "key_kpi": key_kpi,
                    "provider": "Groq",
                }
            except DataLoadError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"RootIQ could not complete the analysis: {exc}")
                return

    results = st.session_state.get("rootiq_results")
    if not results:
        st.markdown("""
        ### How RootIQ works
        **Upload → Profile → Analyze → Find evidence → Identify problems → Explore likely causes → Recommend next actions**

        RootIQ uses Python/Pandas for quantitative analysis first. The AI then interprets the
        computed evidence instead of receiving the entire raw dataset.
        """)
        return

    if results.get("ai_error"):
        st.warning("AI interpretation was unavailable, but the evidence-based Python analysis is still shown below.")
        with st.expander("AI setup/error details"):
            st.code(results["ai_error"])
    if results.get("raw_ai"):
        with st.expander("Raw AI output (JSON parsing fallback)"):
            st.code(results["raw_ai"])

    render_dashboard(results)

    st.divider()
    st.subheader("💬 Ask RootIQ about this dataset")
    st.caption("Answers are restricted to the uploaded dataset and RootIQ's computed findings.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about the uploaded data...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Checking the uploaded evidence..."):
                try:
                    answer = chat_with_dataset(
                        question,
                        results["payload"],
                        st.session_state.chat_history,
                    )
                except AIProviderError as exc:
                    answer = f"I couldn't access the AI provider: {exc}"
                st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
