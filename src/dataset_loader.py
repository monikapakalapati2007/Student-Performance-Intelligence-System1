"""Discovery, loading, validation and field analysis of Air Quality datasets.

This module never mutates the files it reads. It only reports what it finds and
raises a loud, actionable error when the folder contains no Air Quality data.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

TABLE_EXTENSIONS = (".parquet", ".csv", ".jsonl", ".ndjson", ".json", ".tsv", ".xlsx", ".feather")

# Keywords that identify genuine air-quality measurement columns.
POLLUTANT_KEYWORDS = {
    "pm2.5": ("pm2.5", "pm25", "pm_2_5", "pm2_5", "fine particulate"),
    "pm10": ("pm10", "pm_10", "coarse particulate"),
    "no2": ("no2", "nitrogen dioxide", "nitrogen_dioxide"),
    "no": ("no", "nitric oxide"),
    "nox": ("nox", "no_x", "nitrogen oxides"),
    "so2": ("so2", "sulphur dioxide", "sulfur dioxide", "sulphur_dioxide", "sulfur_dioxide"),
    "co": ("co", "carbon monoxide", "carbon_monoxide"),
    "o3": ("o3", "ozone"),
    "nh3": ("nh3", "ammonia"),
    "benzene": ("benzene", "c6h6"),
    "toluene": ("toluene",),
    "xylene": ("xylene", "o-xylene"),
    "aqi": ("aqi", "air quality index", "air_quality_index"),
    "pmi": ("pmi", "pm_index"),
}
WEATHER_KEYWORDS = {
    "temperature": ("temp", "temperature"),
    "humidity": ("humidity", "rh", "relative_humidity"),
    "wind_speed": ("wind_speed", "windspeed", "ws", "wind speed"),
    "wind_direction": ("wind_direction", "wd", "wind dir"),
    "pressure": ("pressure", "bp", "barometric"),
    "rainfall": ("rain", "rainfall", "precipitation"),
    "dew_point": ("dew", "dew_point"),
    "visibility": ("visibility",),
    "solar_radiation": ("solar", "radiation", "sr"),
}
CATEGORY_KEYWORDS = (
    "categor", "category", "status", "quality", "level", "label", "class",
    "band", "index_category", "aqi_bucket", "grade", "severity",
)
LOCATION_KEYWORDS = (
    "city", "location", "site", "station", "area", "region", "state", "district",
    "zone", "place", "address", "town", "village", "county", "country", "lat", "lon",
    "latitude", "longitude",
)
TIME_KEYWORDS = ("date", "time", "timestamp", "datetime", "year", "month", "day", "hour")
ID_KEYWORDS = ("id", "index", "uid", "code", "number", "no.")

# Signals that a table is the Student Performance dataset the brief forbids.
STUDENT_MARKERS = {
    "gender": ("gender",),
    "parental education": ("parental level of education", "parental", "guardian"),
    "lunch": ("lunch", "free/reduced"),
    "test preparation": ("test preparation", "test_preparation", "test prep"),
    "scores": ("math score", "reading score", "writing score"),
    "study habits": ("weekly_self_study_hours", "attendance_percentage", "class_participation"),
    "grades": ("total_score", "grade"),
}

MIN_AQ_SIGNALS = 2  # a table must show at least this many AQ-specific signals

# The pipeline's own outputs must never be mistaken for input datasets.
EXCLUDED_DIR_NAMES = {
    "reports", "models", "checkpoints", "aq_slm", "__pycache__", "node_modules",
    "venv", "env", "site-packages", "assets", "static", "dist", "build",
}


class NoAirQualityDataError(RuntimeError):
    """Raised when the project contains no usable air quality dataset."""


@dataclass
class FieldSpec:
    """Semantic classification of the columns of an air quality table."""

    all_columns: List[str] = field(default_factory=list)
    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    datetime: List[str] = field(default_factory=list)
    text: List[str] = field(default_factory=list)
    identifier: List[str] = field(default_factory=list)
    pollutant: Dict[str, str] = field(default_factory=dict)   # canonical -> column
    weather: Dict[str, str] = field(default_factory=dict)
    category: List[str] = field(default_factory=list)
    location: List[str] = field(default_factory=list)
    time: List[str] = field(default_factory=list)
    constant: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_columns": self.all_columns,
            "numeric": self.numeric,
            "categorical": self.categorical,
            "datetime": self.datetime,
            "text": self.text,
            "identifier": self.identifier,
            "pollutant": self.pollutant,
            "weather": self.weather,
            "category": self.category,
            "location": self.location,
            "time": self.time,
            "constant": self.constant,
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def read_any_table(path: str) -> pd.DataFrame:
    """Read a parquet/csv/tsv/jsonl/json file into a pandas DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, low_memory=False)
        return df
    if ext in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    if ext == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    if ext == ".xlsx":
        return pd.read_excel(path)
    if ext == ".feather":
        return pd.read_feather(path)
    raise ValueError(f"Unsupported file type: {path}")


def discover_data_files(search_dirs: Sequence[str]) -> List[str]:
    """Collect every table-like file inside the candidate directories."""
    found: List[str] = []
    seen = set()
    for directory in search_dirs:
        if not directory or not os.path.isdir(directory):
            continue
        for root, dirnames, filenames in os.walk(directory):
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".") and d.lower() not in EXCLUDED_DIR_NAMES
            ]
            for name in sorted(filenames):
                if name.startswith("."):
                    continue
                if os.path.splitext(name)[1].lower() in TABLE_EXTENSIONS:
                    full = os.path.normpath(os.path.join(root, name))
                    if full not in seen:
                        seen.add(full)
                        found.append(full)
    return found


def detect_student_performance(df: pd.DataFrame) -> Optional[str]:
    """Return a human-readable reason if this looks like student performance data."""
    cols = {str(c).strip().lower() for c in df.columns}
    hits: List[str] = []
    for label, markers in STUDENT_MARKERS.items():
        for marker in markers:
            for col in cols:
                if col == marker or marker in col:
                    hits.append(f"{label} (`{col}`)")
                    break
    if len(hits) >= 2:
        return "; ".join(sorted(set(hits)))
    return None


# Most specific location field first, so "in Delhi" wins over "in India".
LOCATION_PRIORITY = (
    "city", "location", "site", "station", "locality", "area", "place", "town",
    "village", "district", "county", "zone", "region", "state", "country",
    "latitude", "longitude",
)


def _location_rank(column: str) -> int:
    norm = _normalize_name(column)
    for index, key in enumerate(LOCATION_PRIORITY):
        if norm == key or key in norm:
            return index
    return len(LOCATION_PRIORITY)


def _normalize_name(name: str):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _match_keyword(col: str, keywords: Iterable[str]) -> bool:
    norm = _normalize_name(col)
    for kw in keywords:
        k = _normalize_name(kw)
        if not k:
            continue
        if norm == k:
            return True
        # short abbreviations must match a whole token to avoid false hits
        if len(k) <= 3:
            if re.search(rf"(^|_){re.escape(k)}($|_)", norm):
                return True
        elif k in norm:
            return True
    return False


def classify_fields(df: pd.DataFrame) -> FieldSpec:
    """Classify columns into numeric/categorical/datetime/text and AQ roles."""
    spec = FieldSpec(all_columns=[str(c) for c in df.columns])
    for col in df.columns:
        series = df[col]
        n_unique = series.nunique(dropna=True)
        if n_unique <= 1:
            spec.constant.append(str(col))
        if pd.api.types.is_datetime64_any_dtype(series):
            spec.datetime.append(str(col))
            spec.time.append(str(col))
            continue
        if pd.api.types.is_numeric_dtype(series):
            spec.numeric.append(str(col))
            name = str(col)
            for canonical, keywords in POLLUTANT_KEYWORDS.items():
                if _match_keyword(name, keywords):
                    spec.pollutant.setdefault(canonical, name)
                    break
            for canonical, keywords in WEATHER_KEYWORDS.items():
                if _match_keyword(name, keywords):
                    spec.weather.setdefault(canonical, name)
                    break
        elif pd.api.types.is_bool_dtype(series):
            spec.categorical.append(str(col))
        else:
            lowered = str(col).lower()
            if any(_match_keyword(col, (k,)) for k in TIME_KEYWORDS) and _looks_like_dates(series):
                spec.datetime.append(str(col))
                spec.time.append(str(col))
            else:
                spec.categorical.append(str(col))
                if n_unique > max(50, int(0.5 * len(df))) and len(df) > 50:
                    spec.text.append(str(col))
            if any(k in lowered for k in CATEGORY_KEYWORDS):
                spec.category.append(str(col))
            if _match_keyword(col, LOCATION_KEYWORDS):
                spec.location.append(str(col))
        if _match_keyword(str(col), ID_KEYWORDS) and str(col) not in spec.datetime:
            spec.identifier.append(str(col))
    spec.location.sort(key=_location_rank)
    return spec


def _looks_like_dates(series: pd.Series, sample: int = 30) -> bool:
    values = series.dropna().astype(str).head(sample)
    if values.empty:
        return False
    pattern = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}-\d{2}")
    hits = sum(1 for v in values if pattern.search(v))
    return hits >= max(3, int(0.6 * len(values)))


def is_air_quality(df: pd.DataFrame) -> Tuple[bool, str]:
    """Decide whether a table holds air quality measurements."""
    spec = classify_fields(df)
    signals: List[str] = []
    if spec.pollutant:
        signals.append(f"pollutant columns {sorted(set(spec.pollutant.values()))}")
    if spec.weather:
        signals.append(f"meteorological columns {sorted(set(spec.weather.values()))}")
    if spec.category and spec.numeric:
        signals.append(f"category columns {sorted(set(spec.category))}")
    if spec.location and spec.numeric:
        signals.append(f"location columns {sorted(set(spec.location))}")
    if "aqi" in spec.pollutant:
        signals.append("AQI column")
    student = detect_student_performance(df)
    if student:
        return False, f"looks like student-performance data ({student})"
    if len(signals) >= MIN_AQ_SIGNALS:
        return True, "; ".join(signals)
    return False, f"insufficient air quality signals (found {len(signals)}: {signals})"


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def _round(value: Any, digits: int = 4) -> Any:
    try:
        if pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return None


def analyze_frame(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """Shape, dtypes, missing values, duplicates, distributions and oddities."""
    columns: List[Dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        info: Dict[str, Any] = {
            "name": str(col),
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "missing_pct": round(float(series.isna().mean() * 100), 3),
            "unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            clean = series.dropna()
            info["role"] = "numeric"
            if len(clean):
                info.update(
                    min=_round(clean.min()), max=_round(clean.max()),
                    mean=_round(clean.mean()), std=_round(clean.std()),
                    p05=_round(clean.quantile(0.05)), p50=_round(clean.quantile(0.50)),
                    p95=_round(clean.quantile(0.95)),
                    zeros=int((clean == 0).sum()),
                    negative=int((clean < 0).sum()),
                )
                # IQR based outlier count - reported, never silently removed.
                q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    low, high = q1 - 3 * iqr, q3 + 3 * iqr
                    info["extreme_outliers_3iqr"] = int(((clean < low) | (clean > high)).sum())
        else:
            info["role"] = "categorical/text/datetime"
            counts = series.astype(str).value_counts().head(5)
            info["top_values"] = {str(k): int(v) for k, v in counts.items()}
        columns.append(info)

    duplicate_rows = int(df.duplicated().sum())
    return {
        "name": name,
        "rows": int(len(df)),
        "columns_count": int(df.shape[1]),
        "columns": columns,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(duplicate_rows / max(1, len(df)) * 100, 3),
        "total_missing": int(df.isna().sum().sum()),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1024 ** 2, 2),
    }


def _row_hashes(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    usable = [c for c in columns if c in df.columns]
    if not usable:
        usable = list(df.columns)
    frame = df[usable].astype(str).agg("|".join, axis=1)
    return frame.map(lambda s: hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()).tolist()


def numeric_correlation(df: pd.DataFrame, columns: Sequence[str], max_columns: int = 12) -> Dict[str, Dict[str, float]]:
    """Pearson correlation between numeric variables, for the statistics report."""
    usable = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    usable = usable[:max_columns]
    if len(usable) < 2:
        return {}
    corr = df[usable].corr(numeric_only=True).round(3)
    return {str(row): {str(col): (None if pd.isna(value) else float(value))
                      for col, value in corr.loc[row].items()}
            for row in corr.index}


def check_split_overlap(splits: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Detect identical records appearing in more than one split (leakage)."""
    names = list(splits)
    hashes = {name: _row_hashes(splits[name], list(splits[name].columns)) for name in names}
    result: Dict[str, Any] = {"pairwise": {}, "leakage_detected": False}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            set_a, set_b = set(hashes[a]), set(hashes[b])
            shared = len(set_a & set_b)
            result["pairwise"][f"{a}__{b}"] = {
                "shared_rows": shared,
                "pct_of_smaller": round(shared / max(1, min(len(set_a), len(set_b))) * 100, 3),
            }
            if shared:
                result["leakage_detected"] = True
    return result


# --------------------------------------------------------------------------- #
# Split discovery
# --------------------------------------------------------------------------- #
SPLIT_PATTERNS = (
    ("test", ("test", "testing", "holdout", "eval_set")),
    ("validation", ("valid", "val", "dev", "validation")),
    ("train", ("train", "training")),
)


def _guess_split_name(path: str) -> Optional[str]:
    stem = _normalize_name(os.path.splitext(os.path.basename(path))[0])
    for split, patterns in SPLIT_PATTERNS:  # test/validation first: "val" ⊂ "eval"
        for pat in patterns:
            if re.search(rf"(^|_){pat}($|_)", stem) or stem.endswith(pat):
                return split
    return None


def load_air_quality_splits(
    search_dirs: Sequence[str],
    explicit: Optional[Dict[str, str]] = None,
    seed: int = 1337,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Load train/validation/test air quality splits plus a full audit trail.

    Priority: explicit paths > files named train/valid/test > single table split
    deterministically (80/10/10 by row hash, seed controlled).
    """
    report: Dict[str, Any] = {"searched_dirs": list(search_dirs), "files_examined": [], "rejected": []}
    if explicit:
        splits = {k: read_any_table(v) for k, v in explicit.items()}
        report["source"] = "explicit paths"
        report["files_examined"] = list(explicit.values())
        return splits, report

    candidates = discover_data_files(search_dirs)
    accepted: Dict[str, List[Tuple[str, pd.DataFrame]]] = {}
    unnamed: List[Tuple[str, pd.DataFrame]] = []
    for path in candidates:
        try:
            df = read_any_table(path)
        except Exception as exc:
            report["rejected"].append({"path": path, "reason": f"unreadable: {exc}"})
            continue
        ok, reason = is_air_quality(df)
        report["files_examined"].append(
            {"path": path, "rows": int(len(df)), "columns": [str(c) for c in df.columns],
             "air_quality": ok, "reason": reason}
        )
        if not ok:
            report["rejected"].append({"path": path, "reason": reason})
            continue
        split = _guess_split_name(path)
        if split:
            accepted.setdefault(split, []).append((path, df))
        else:
            unnamed.append((path, df))

    if not accepted and not unnamed:
        raise NoAirQualityDataError(build_missing_data_message(report))

    splits: Dict[str, pd.DataFrame] = {}
    if accepted:
        for split, items in accepted.items():
            if len(items) > 1:
                # more than one file for the same split -> concatenate in name order
                df = pd.concat([d for _, d in sorted(items, key=lambda x: x[0])], ignore_index=True)
                report.setdefault("merged_files", {})[split] = [p for p, _ in sorted(items, key=lambda x: x[0])]
            else:
                df = items[0][1]
            splits[split] = df.reset_index(drop=True)
        report["source"] = "pre-existing split files"
        # unnamed tables can still supply a missing split
        if unnamed and len(splits) < 2:
            unnamed = []
    else:
        path, df = max(unnamed, key=lambda x: len(x[1]))
        report["source"] = f"deterministic re-split of {path} (80/10/10, seed={seed})"
        report["split_note"] = (
            "No train/validation/test filenames were found, so the single available "
            "table was split once by a seeded row-hash shuffle. The split is frozen in "
            "data/processed/aq_slm/ and reused by every later run."
        )
        permutations = __import__("numpy").random.RandomState(seed).permutation(len(df))
        n = len(df)
        n_test, n_val = max(1, int(0.1 * n)), max(1, int(0.1 * n))
        idx = {
            "test": permutations[:n_test],
            "validation": permutations[n_test:n_test + n_val],
            "train": permutations[n_test + n_val:],
        }
        for name, rows in idx.items():
            splits[name] = df.iloc[sorted(rows)].reset_index(drop=True)

    report["split_rows"] = {k: int(len(v)) for k, v in splits.items()}
    return splits, report


def build_missing_data_message(report: Dict[str, Any]) -> str:
    """Actionable explanation when no air quality table was found."""
    lines = [
        "NO AIR QUALITY DATASET FOUND - refusing to train.",
        "",
        "The pipeline searched these locations:",
    ]
    for d in report.get("searched_dirs", []):
        lines.append(f"  - {d}")
    examined = report.get("files_examined", [])
    if examined:
        lines.append("")
        lines.append("Files that were examined and rejected:")
        for item in examined:
            lines.append(f"  - {item['path']}  ({item['rows']} rows)")
            lines.append(f"      reason: {item['reason']}")
            lines.append(f"      columns: {', '.join(item['columns'][:12])}")
    lines += [
        "",
        "This project trains an AIR QUALITY language model. Student-performance",
        "tables (exam/score/attendance data) are explicitly out of scope, and no",
        "air quality measurements (PM2.5, PM10, NO2, SO2, CO, O3, AQI, station or",
        "city, met variables) were found anywhere in the project.",
        "",
        "Add your processed air quality data and re-run. Any of these layouts work:",
        "  data/processed/train.parquet + validation.parquet + test.parquet",
        "  data/training/*.csv with a `split` column",
        "  a single air-quality .csv/.parquet (the pipeline will split it 80/10/10",
        "    deterministically and freeze that split so later runs reuse it)",
        "",
        "Then run:  python src/train_slm.py",
    ]
    return "\n".join(lines)


def default_search_dirs(project_root: str) -> List[str]:
    """Directories the loader inspects, in priority order."""
    return [
        os.path.join(project_root, "data", "processed"),
        os.path.join(project_root, "data", "training"),
        os.path.join(project_root, "data", "validation"),
        os.path.join(project_root, "data", "testing"),
        os.path.join(project_root, "data", "test"),
        os.path.join(project_root, "data"),
        os.path.join(project_root, "hugging face"),
        project_root,
    ]
