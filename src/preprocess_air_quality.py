"""Air Quality preprocessing: cleaning, text serialisation, QA generation.

Design rules enforced here:
  * only fields that genuinely exist in the table are serialised,
  * no invented measurements, no imputed values written into text,
  * no statistic is learned across splits (nothing to leak: all transformations
    are per-value and stateless),
  * original dataset files are never modified; outputs go to a new folder.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from dataset_loader import FieldSpec, classify_fields

# --------------------------------------------------------------------------- #
# Domain reference text (general knowledge, not data). Used only in the train
# corpus so the model can answer "what does PM2.5 mean" style questions.
# --------------------------------------------------------------------------- #
POLLUTANT_INFO: Dict[str, Dict[str, Any]] = {
    "pm2.5": {
        "label": "PM2.5",
        "unit": "micrograms per cubic metre",
        "definition": (
            "PM2.5 is fine particulate matter with a diameter of 2.5 micrometres or "
            "less. These particles can travel deep into the lungs and into the "
            "bloodstream, so PM2.5 is the pollutant most strongly linked to "
            "cardiovascular and respiratory harm."
        ),
    },
    "pm10": {
        "label": "PM10",
        "unit": "micrograms per cubic metre",
        "definition": (
            "PM10 is particulate matter with a diameter of 10 micrometres or less. "
            "It includes dust, pollen, mould and smoke, and it is coarser than PM2.5. "
            "PM10 irritates the upper airways and is usually higher than PM2.5 in the "
            "same air sample."
        ),
    },
    "no2": {
        "label": "NO2",
        "unit": "micrograms per cubic metre",
        "definition": (
            "NO2 is nitrogen dioxide, a reddish-brown gas produced mainly by burning "
            "fuel in vehicles and power plants. It is a precursor of ground level "
            "ozone and of nitrate particles, and it peaks during morning and evening "
            "traffic."
        ),
    },
    "no": {
        "label": "NO",
        "unit": "micrograms per cubic metre",
        "definition": (
            "NO is nitric oxide, a colourless gas emitted by combustion. It quickly "
            "reacts with ozone to form NO2, so NO is usually highest close to traffic "
            "and is often reported together with NO2 as NOx."
        ),
    },
    "nox": {
        "label": "NOx",
        "unit": "micrograms per cubic metre",
        "definition": (
            "NOx is the sum of nitric oxide and nitrogen dioxide. It is a marker of "
            "traffic and combustion emissions."
        ),
    },
    "so2": {
        "label": "SO2",
        "unit": "micrograms per cubic metre",
        "definition": (
            "SO2 is sulphur dioxide, a sharp-smelling gas released by burning coal and "
            "oil, by smelters and by volcanoes. It forms sulphate particles and acid "
            "rain and aggravates asthma."
        ),
    },
    "co": {
        "label": "CO",
        "unit": "milligrams per cubic metre",
        "definition": (
            "CO is carbon monoxide, a colourless, odourless gas from incomplete "
            "combustion. It binds to haemoglobin and reduces the oxygen carried by the "
            "blood."
        ),
    },
    "o3": {
        "label": "O3",
        "unit": "micrograms per cubic metre",
        "definition": (
            "O3 is ozone. Ground level ozone is not emitted directly: sunlight cooks "
            "NOx together with volatile organic compounds, so ozone usually peaks on "
            "hot, sunny afternoons downwind of traffic."
        ),
    },
    "nh3": {
        "label": "NH3",
        "unit": "micrograms per cubic metre",
        "definition": (
            "NH3 is ammonia, mainly released by agriculture and waste. It neutralises "
            "acid gases and contributes to secondary particulate matter."
        ),
    },
    "benzene": {
        "label": "Benzene",
        "unit": "micrograms per cubic metre",
        "definition": (
            "Benzene is a volatile organic compound present in petrol. It is a known "
            "human carcinogen and a marker of traffic emissions."
        ),
    },
    "toluene": {
        "label": "Toluene",
        "unit": "micrograms per cubic metre",
        "definition": (
            "Toluene is a volatile organic compound used as a solvent and present in "
            "fuel. It contributes to photochemical smog."
        ),
    },
    "xylene": {
        "label": "Xylene",
        "unit": "micrograms per cubic metre",
        "definition": (
            "Xylene is an aromatic solvent emitted by industry and traffic and a "
            "precursor of ground level ozone."
        ),
    },
    "aqi": {
        "label": "AQI",
        "unit": "",
        "definition": (
            "The Air Quality Index, or AQI, is a single number that maps pollutant "
            "concentrations onto a common scale. It is computed from the worst "
            "pollutant sub-index, so the AQI is driven by whichever pollutant is most "
            "elevated at that moment."
        ),
    },
    "pmi": {
        "label": "PM index",
        "unit": "",
        "definition": (
            "A particulate matter index summarises several particle size fractions in "
            "one number."
        ),
    },
}

GENERAL_KNOWLEDGE: List[str] = [
    (
        "Air quality is described by the concentration of pollutants in the air and "
        "by how those concentrations compare with health based guideline values. "
        "The pollutants most often monitored are particulate matter, nitrogen "
        "dioxide, sulphur dioxide, carbon monoxide, ozone and ammonia."
    ),
    (
        "Meteorology controls air pollution. High wind speed disperses pollutants and "
        "lowers concentrations. High humidity and low wind speed let particles "
        "accumulate. Rain washes particles out of the air. High temperature and strong "
        "sunlight accelerate ozone formation."
    ),
    (
        "Particulate matter is described by mass concentration. PM2.5 counts only the "
        "fine fraction, while PM10 counts everything below ten micrometres, so PM10 "
        "is normally the larger number for the same air sample."
    ),
    (
        "An air quality category is a verbal band attached to an index value, for "
        "example good, satisfactory, moderate, poor, very poor or severe. The exact "
        "thresholds depend on the national index that produced the label, so a "
        "category should always be read together with the index it came from."
    ),
    (
        "Pollutant concentrations are reported for a stated averaging period, usually "
        "an hour, eight hours or twenty four hours. A one hour value and a twenty four "
        "hour value for the same place are not directly interchangeable."
    ),
]


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def pretty_label(column: str) -> str:
    text = str(column).replace("_", " ").replace("-", " ").strip()
    text = re.sub(r"\s+", " ", text)
    known = {
        "pm2 5": "PM2.5", "pm25": "PM2.5", "pm2.5": "PM2.5", "pm10": "PM10",
        "no2": "NO2", "no": "NO", "nox": "NOx", "so2": "SO2", "co": "CO",
        "o3": "O3", "nh3": "NH3", "aqi": "AQI", "rh": "relative humidity",
    }
    low = text.lower()
    if low in known:
        return known[low]
    if low.isupper() and len(low) <= 4:
        return text.upper()
    return text[:1].upper() + text[1:]


def unit_for(column: str, spec: FieldSpec) -> str:
    profile = _profile()
    if profile is not None and column in profile.units:
        return profile.units[column]
    for canonical, col in spec.pollutant.items():
        if col == column:
            return POLLUTANT_INFO.get(canonical, {}).get("unit", "")
    lowered = str(column).lower()
    if "temp" in lowered:
        return "degrees Celsius"
    if "humid" in lowered or lowered == "rh":
        return "%"
    if "wind" in lowered and "speed" in lowered:
        return "metres per second"
    if "pressure" in lowered:
        return "hectopascal"
    if "rain" in lowered or "precip" in lowered:
        return "millimetres"
    return ""


def format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if number != number:  # NaN
        return "not recorded"
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    text = f"{number:.4f}".rstrip("0").rstrip(".")
    return text


def format_value(value: Any, column: str, spec: FieldSpec) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "not recorded"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = format_number(value)
        unit = unit_for(column, spec)
        return f"{number} {unit}".strip()
    text = str(value).strip()
    if text == "" or text.lower() in ("nan", "none", "null", "na", "n/a"):
        return "not recorded"
    return text


# --------------------------------------------------------------------------- #
# Cleaning (stateless, per-value; fitted statistics are deliberately not used)
# --------------------------------------------------------------------------- #
def clean_frame(df: pd.DataFrame, spec: FieldSpec) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Light, documented cleaning that cannot leak information across splits."""
    log: Dict[str, Any] = {"steps": []}
    out = df.copy()

    # 1. strip whitespace / empty strings in object columns
    stripped = 0
    for col in out.columns:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            before = out[col].copy()
            out[col] = out[col].astype("string").str.strip()
            out[col] = out[col].replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
            stripped += int((before.astype("string") != out[col]).sum())
    log["steps"].append(f"stripped whitespace and empty strings in text columns ({stripped} values normalised)")

    # 2. coerce numeric-looking object columns
    coerced = []
    for col in out.columns:
        if col in spec.numeric or out[col].dtype != object:
            continue
        as_str = out[col].astype("string").str.replace(",", "", regex=False)
        numeric = pd.to_numeric(as_str, errors="coerce")
        non_null = out[col].notna().sum()
        if non_null and numeric.notna().sum() >= 0.9 * non_null:
            out[col] = numeric
            coerced.append(str(col))
    if coerced:
        log["steps"].append(f"coerced numeric-looking text columns: {coerced}")

    # 3. drop exact duplicate rows inside this split only
    before_rows = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    log["duplicate_rows_removed"] = before_rows - len(out)
    log["steps"].append(
        f"dropped {before_rows - len(out)} exact duplicate rows within this split "
        "(duplicates are never removed across splits, and splits are never re-shuffled)"
    )

    # 4. parse timestamps (no imputation)
    for col in spec.datetime:
        if col in out.columns and not pd.api.types.is_datetime64_any_dtype(out[col]):
            parsed = pd.to_datetime(out[col], errors="coerce", format="mixed", dayfirst=False)
            if parsed.notna().sum():
                out[col] = parsed
                log["steps"].append(f"parsed `{col}` as datetime")
    log["rows_after"] = int(len(out))
    log["rows_before"] = int(before_rows)
    return out, log


# --------------------------------------------------------------------------- #
# Prompt scaffolding shared by preprocessing, evaluation and inference
# --------------------------------------------------------------------------- #
Q_PREFIX = "Question:"
A_PREFIX = "Answer:"
RECORD_START = "### Air quality record"
QA_START = "### Question and answer"
DOC_END = "### End"

# The active dataset profile supplies domain wording, units and reference text.
# A module-level handle keeps the many formatting helpers simple; it is set once
# by prepare() at the start of a run.
_ACTIVE_PROFILE = None


def set_profile(profile) -> None:
    global _ACTIVE_PROFILE
    _ACTIVE_PROFILE = profile


def _profile():
    return _ACTIVE_PROFILE


def _context(row: pd.Series, spec: FieldSpec) -> str:
    """Short locator phrase: location + timestamp, from fields that exist."""
    parts: List[str] = []
    for col in spec.location[:2]:
        if col in row.index:
            value = format_value(row[col], col, spec)
            if value != "not recorded" and value.replace(".", "").isdigit() is False:
                parts.append(f"in {value}")
                break
    for col in spec.time[:2]:
        if col in row.index:
            value = row[col]
            if pd.notna(value):
                try:
                    parts.append(f"on {pd.Timestamp(value).strftime('%Y-%m-%d %H:%M')}")
                except Exception:
                    parts.append(f"on {format_value(value, col, spec)}")
                break
    if not parts:
        # fall back to an identifier column (e.g. student_id) or the row position
        for col in spec.identifier:
            if col in row.index and pd.notna(row[col]):
                label = (_profile().identity_label if _profile() else "record")
                parts.append(f"for {label} {format_value(row[col], col, spec)}")
                break
    if not parts:
        parts.append(f"record {row.name}")
    return " ".join(parts)


# Index fields summarise several pollutants; they are not themselves pollutants.
INDEX_KEYS = {"aqi", "pmi"}


def _pollutant_pairs(row: pd.Series, spec: FieldSpec) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for canonical, column in spec.pollutant.items():
        if canonical in INDEX_KEYS or column not in row.index:
            continue
        value = row[column]
        if pd.isna(value):
            continue
        label = POLLUTANT_INFO.get(canonical, {}).get("label", pretty_label(column))
        pairs.append((label, format_value(value, column, spec)))
    return pairs


def _index_pairs(row: pd.Series, spec: FieldSpec) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for canonical, column in spec.pollutant.items():
        if canonical not in INDEX_KEYS or column not in row.index or pd.isna(row[column]):
            continue
        label = POLLUTANT_INFO.get(canonical, {}).get("label", pretty_label(column))
        pairs.append((label, format_value(row[column], column, spec)))
    return pairs


def _weather_pairs(row: pd.Series, spec: FieldSpec) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for _, column in spec.weather.items():
        if column not in row.index or pd.isna(row[column]):
            continue
        pairs.append((pretty_label(column), format_value(row[column], column, spec)))
    return pairs


def _category_value(row: pd.Series, spec: FieldSpec) -> Optional[str]:
    for column in spec.category:
        if column in row.index and pd.notna(row[column]):
            return format_value(row[column], column, spec)
    return None


def row_card(row: pd.Series, spec: FieldSpec) -> str:
    """Serialise one measurement row as a readable record card."""
    context = _context(row, spec)
    heading = _profile().record_heading if _profile() else RECORD_START
    lines = [f"{heading}{': ' + context if context else ''}."]

    pollutants = _pollutant_pairs(row, spec)
    if pollutants:
        lines.append("Pollutants: " + "; ".join(f"{k} {v}" for k, v in pollutants) + ".")
    for label, value in _index_pairs(row, spec):
        lines.append(f"Air quality index: {label} {value}.")
    weather = _weather_pairs(row, spec)
    if weather:
        lines.append("Weather: " + "; ".join(f"{k} {v}" for k, v in weather) + ".")
    category = _category_value(row, spec)
    if category:
        category_label = _profile().category_label if _profile() else "Air quality category"
        lines.append(f"{category_label}: {category}.")

    # any remaining informative numeric columns not already serialised
    covered = (set(spec.pollutant.values()) | set(spec.weather.values())
               | set(spec.category) | set(spec.identifier))
    extras = [c for c in spec.numeric if c not in covered]
    extra_parts = [
        f"{pretty_label(c)} {format_value(row[c], c, spec)}"
        for c in extras if c in row.index and pd.notna(row[c])
    ]
    if extra_parts:
        label = "Other measurements" if (pollutants or weather) else "Measurements"
        lines.append(f"{label}: " + "; ".join(extra_parts) + ".")
    for col in spec.categorical:
        if col in spec.category or col in spec.location or col not in row.index:
            continue
        value = row[col]
        if pd.notna(value) and str(value).strip():
            lines.append(f"{pretty_label(col)}: {str(value).strip()}.")
    lines.append(DOC_END)
    return " ".join(lines)


def qa_pairs_for_row(row: pd.Series, spec: FieldSpec, max_pairs: int = 4) -> List[Tuple[str, str]]:
    """Deterministic question/answer pairs whose answers come only from the row."""
    context = _context(row, spec)
    located = f" {context}" if context else ""
    candidates: List[Tuple[str, str]] = []

    pollutants = _pollutant_pairs(row, spec)
    if pollutants:
        label, value = pollutants[0]
        candidates.append((f"What is the {label} level{located}?", f"The {label} level is {value}."))
        names = ", ".join(k for k, _ in pollutants)
        candidates.append(
            (f"Which pollutants were measured{located}?", f"The measured pollutants are {names}.")
        )
    category = _category_value(row, spec)
    if category:
        category_label = _profile().category_label if _profile() else "Air quality category"
        candidates.append(
            (f"What is the {category_label.lower()}{located}?",
             f"The {category_label.lower()} is {category}.")
        )
    for label, value in _index_pairs(row, spec):
        candidates.append((f"What is the {label}{located}?", f"The {label} is {value}."))

    # Field-by-field questions for the remaining numeric measurements. These make
    # any tabular schema learnable (student scores and study hours as well as
    # pollutant concentrations).
    covered = (set(spec.pollutant.values()) | set(spec.weather.values())
               | set(spec.category) | set(spec.identifier))
    for column in spec.numeric:
        if column in covered or column not in row.index or pd.isna(row[column]):
            continue
        label = pretty_label(column)
        value = format_value(row[column], column, spec)
        candidates.append((f"What is the {label.lower()}{located}?", f"The {label.lower()} is {value}."))
    weather = _weather_pairs(row, spec)
    if weather:
        desc = "; ".join(f"{k} {v}" for k, v in weather)
        candidates.append((f"What were the weather conditions{located}?", f"The weather conditions were {desc}."))

    # summary derived strictly from present fields
    bits: List[str] = []
    if pollutants:
        bits.append("pollutants " + ", ".join(f"{k} {v}" for k, v in pollutants))
    if weather:
        bits.append("weather " + ", ".join(f"{k} {v}" for k, v in weather))
    if category:
        bits.append(f"air quality category {category}")
    if bits:
        candidates.append(
            (f"Summarise the air quality information{located}.",
             "This record reports " + "; ".join(bits) + ".")
        )
    return candidates[:max_pairs]


def qa_document(question: str, answer: str, context: str = "") -> str:
    """One QA document, optionally grounded on the record card it refers to.

    Grounded examples teach the model to READ a value out of the context instead of
    recalling it from parameters, which is what makes record-specific questions
    answerable at this model size.
    """
    body = f"{QA_START}\n"
    if context:
        body += f"{context}\n"
    body += f"{Q_PREFIX} {question}\n{A_PREFIX} {answer}\n{DOC_END}"
    return body


def record_context(row: pd.Series, spec: FieldSpec) -> str:
    """The row card without its end marker, for use as prompt context."""
    card = row_card(row, spec)
    suffix = f" {DOC_END}"
    return card[:-len(suffix)] if card.endswith(suffix) else card


def knowledge_documents(columns: Sequence[str], spec: FieldSpec) -> List[str]:
    """Reference text for the domain of the active profile."""
    profile = _profile()
    if profile is not None and profile.key != "air_quality":
        header = f"### {profile.subject.title()} knowledge"
        return [f"{header}\n{text}\n{DOC_END}" for text in profile.knowledge]
    docs = [f"### Air quality knowledge\n{text}\n{DOC_END}" for text in GENERAL_KNOWLEDGE]
    if profile is not None:
        docs += [f"### Air quality knowledge\n{text}\n{DOC_END}" for text in profile.knowledge]
    for canonical, column in spec.pollutant.items():
        info = POLLUTANT_INFO.get(canonical)
        if not info:
            continue
        docs.append(f"### Air quality knowledge\n{info['label']}: {info['definition']}\n{DOC_END}")
    return docs


def dataset_fact_documents(spec: FieldSpec, train_rows: int) -> List[str]:
    """Dataset-level facts computed from the training split only."""
    docs: List[str] = []
    labels = [
        POLLUTANT_INFO.get(c, {}).get("label", pretty_label(col))
        for c, col in spec.pollutant.items()
    ]
    if labels:
        docs.append(qa_document(
            "Which pollutants are present in the dataset?",
            f"The dataset contains measurements of {', '.join(labels)}.",
        ))
    if spec.location:
        docs.append(qa_document(
            "Which fields describe where a measurement was taken?",
            f"Location is described by {', '.join(pretty_label(c) for c in spec.location)}.",
        ))
    if spec.weather:
        docs.append(qa_document(
            "Which meteorological variables are recorded?",
            "The recorded meteorological variables are "
            + ", ".join(pretty_label(c) for c in spec.weather.values()) + ".",
        ))
    if spec.category:
        cat = _profile().category_label if _profile() else "air quality category"
        docs.append(qa_document(
            f"Which field gives the {cat.lower()}?",
            f"The {cat.lower()} is given by {', '.join(pretty_label(c) for c in spec.category)}.",
        ))
    docs.append(qa_document(
        "What does the dataset describe?",
        f"The dataset holds {train_rows} air quality observations, each one recording "
        "pollutant concentrations together with location, time and weather context.",
    ))
    return docs


# --------------------------------------------------------------------------- #
# Corpus construction
# --------------------------------------------------------------------------- #
def build_documents(
    df: pd.DataFrame,
    spec: FieldSpec,
    include_knowledge: bool = False,
    max_qa_per_row: int = 4,
    train_rows: int = 0,
) -> List[str]:
    """Turn a table into text documents (record cards + grounded QA pairs)."""
    """Turn a table into a list of text documents."""
    docs: List[str] = []
    if include_knowledge:
        docs.extend(dataset_fact_documents(spec, train_rows or len(df)))
        docs.extend(knowledge_documents(list(df.columns), spec))
    for _, row in df.iterrows():
        docs.append(row_card(row, spec))
        context = record_context(row, spec)
        for question, answer in qa_pairs_for_row(row, spec, max_qa_per_row):
            docs.append(qa_document(question, answer, context=context))
    return docs


def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    import json

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare(
    splits: Dict[str, pd.DataFrame],
    output_dir: str,
    spec: Optional[FieldSpec] = None,
    profile=None,
) -> Dict[str, Any]:
    """Clean every split, serialise to text and persist the processed corpus.

    Returns a report dict with sizes, cleaning logs and example documents.
    """
    from dataset_loader import analyze_frame

    if profile is not None:
        set_profile(profile)
    spec = spec or classify_fields(splits[list(splits)[0]])
    report: Dict[str, Any] = {"output_dir": output_dir, "splits": {}, "field_spec": spec.to_dict()}
    cleaned: Dict[str, pd.DataFrame] = {}
    train_rows = len(splits.get("train", next(iter(splits.values()))))

    for name, df in splits.items():
        clean_df, log = clean_frame(df, spec)
        cleaned[name] = clean_df
        docs = build_documents(
            clean_df, spec,
            include_knowledge=(name == "train"),
            train_rows=train_rows if name == "train" else 0,
        )
        records = [{"id": i, "text": text, "split": name} for i, text in enumerate(docs)]
        path = os.path.join(output_dir, f"{name}.jsonl")
        write_jsonl(records, path)
        stats = analyze_frame(clean_df, f"{name}_cleaned")
        report["splits"][name] = {
            "source_rows": int(len(df)),
            "rows_after_cleaning": int(len(clean_df)),
            "documents": len(docs),
            "characters": int(sum(len(d) for d in docs)),
            "path": path,
            "cleaning": log,
            "duplicate_rows_after_cleaning": stats["duplicate_rows"],
        }
        cleaned[name] = clean_df

    train_split = "train" if "train" in cleaned else list(cleaned)[0]
    example_spec = spec
    example_row = cleaned[train_split].iloc[0]
    report["examples"] = {
        "record_card": row_card(example_row, example_spec),
        "qa_pairs": [
            {"question": q, "answer": a}
            for q, a in qa_pairs_for_row(example_row, example_spec)
        ],
    }
    report["total_documents"] = int(sum(v["documents"] for v in report["splits"].values()))
    report["total_characters"] = int(sum(v["characters"] for v in report["splits"].values()))
    return report, cleaned, spec
