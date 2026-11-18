"""Dataset profiles: which tabular domain this pipeline is modelling.

The pipeline was written for air quality records. A second profile,
``student_performance``, is provided for the Kaggle "student_performance 1
million" table. A profile supplies everything domain-specific: which columns
identify the dataset, how records are named, what domain knowledge is added to
the training corpus, and what the reports and paper are titled.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple


@dataclass
class DatasetProfile:
    key: str
    display_name: str
    subject: str
    record_heading: str
    paper_title: str
    model_dirname: str
    markers: Dict[str, Tuple[str, ...]]
    min_markers: int
    units: Dict[str, str] = field(default_factory=dict)
    knowledge: List[str] = field(default_factory=list)
    identity_label: str = "record"
    category_label: str = "Air quality category"
    description: str = ""
    questions: List[str] = field(default_factory=list)

    def marker_columns(self) -> Dict[str, List[str]]:
        return {label: list(keys) for label, keys in self.markers.items()}


AIR_QUALITY = DatasetProfile(
    key="air_quality",
    display_name="Air Quality SLM",
    subject="air quality",
    record_heading="### Air quality record",
    paper_title=("A Compact Decoder-Only Small Language Model for Air Quality "
                 "Record Understanding: Design, Training and Evaluation"),
    model_dirname="air_quality_slm",
    markers={
        "pollutant": ("pm2.5", "pm25", "pm10", "no2", "so2", "co", "o3", "aqi",
                      "nh3", "benzene", "nox"),
        "meteorology": ("temperature", "temp", "humidity", "wind_speed", "pressure",
                        "rain", "dew", "visibility"),
        "category": ("aqi_category", "category", "status", "quality", "band"),
        "location": ("city", "station", "site", "location", "state", "district"),
    },
    min_markers=2,
    knowledge=[
        "Air quality is described by the concentration of pollutants in the air and by how "
        "those concentrations compare with health based guideline values. The pollutants most "
        "often monitored are particulate matter, nitrogen dioxide, sulphur dioxide, carbon "
        "monoxide, ozone and ammonia.",
        "Meteorology controls air pollution. High wind speed disperses pollutants and lowers "
        "concentrations. High humidity and low wind speed let particles accumulate. Rain washes "
        "particles out of the air.",
        "Particulate matter is described by mass concentration. PM2.5 counts only the fine "
        "fraction, while PM10 counts everything below ten micrometres, so PM10 is normally the "
        "larger number for the same air sample.",
    ],
    identity_label="record",
    category_label="Air quality category",
    description="air quality measurements (pollutants, meteorology, index, location)",
    questions=[
        "What does PM2.5 represent?",
        "Which pollutants are present in the dataset?",
        "Which meteorological variables are recorded?",
        "Explain the relationship between particulate matter and weather conditions.",
        "What does the Air Quality Index summarise?",
    ],
)

STUDENT_PERFORMANCE = DatasetProfile(
    key="student_performance",
    display_name="Student Performance SLM",
    subject="student performance",
    record_heading="### Student record",
    paper_title=("A Compact Decoder-Only Small Language Model for Student Performance "
                 "Record Understanding: Design, Training and Evaluation"),
    model_dirname="student_performance_slm",
    markers={
        "study behaviour": ("weekly_self_study_hours", "self_study", "study_hours"),
        "attendance": ("attendance_percentage", "attendance"),
        "participation": ("class_participation", "participation"),
        "score": ("total_score", "math score", "reading score", "writing score", "score"),
        "grade": ("grade", "final_grade", "letter_grade"),
        "demographics": ("gender", "parental level of education", "lunch",
                         "test preparation course", "race/ethnicity"),
    },
    min_markers=2,
    units={
        "weekly_self_study_hours": "hours per week",
        "attendance_percentage": "%",
        "class_participation": "points out of 10",
        "total_score": "points out of 100",
    },
    knowledge=[
        "This corpus contains student performance records. Each record describes one student "
        "by study behaviour, attendance, class participation, a final score and a letter grade.",
        "Weekly self study hours record how many hours a student studies outside class in a "
        "typical week; the values in this dataset range from 0 to 40 hours.",
        "Attendance percentage is the share of scheduled classes a student actually attended, "
        "expressed between 0 and 100 percent.",
        "Class participation is a score between 0 and 10 describing how actively a student "
        "takes part in class.",
        "Total score is the final mark out of 100. The letter grade is derived from the total "
        "score: A is the highest band, then B, C and D, and F indicates failure.",
        "Study behaviour, attendance and participation are all expected to move together with "
        "the final score: students who attend more and participate more tend to score higher.",
    ],
    identity_label="student",
    category_label="Grade",
    description="student performance (study hours, attendance, participation, score, grade)",
    questions=[
        "What does the grade represent?",
        "Which study behaviour fields are recorded?",
        "Explain the relationship between attendance and total score.",
        "Summarise a student record.",
        "What is the range of weekly self study hours?",
    ],
)

PROFILES: Dict[str, DatasetProfile] = {
    AIR_QUALITY.key: AIR_QUALITY,
    STUDENT_PERFORMANCE.key: STUDENT_PERFORMANCE,
}


def get_profile(key: str) -> DatasetProfile:
    if key == "auto":
        raise ValueError("resolve 'auto' with detect_profile() before calling get_profile()")
    if key not in PROFILES:
        raise ValueError(f"Unknown profile '{key}'. Options: {sorted(PROFILES)} or 'auto'")
    return PROFILES[key]


def all_profiles() -> Sequence[DatasetProfile]:
    return list(PROFILES.values())


def default_models_dir(project_root: str) -> str:
    """Locate the trained model directory, whichever profile produced it."""
    root = os.path.join(project_root, "models")
    preferred = os.path.join(root, AIR_QUALITY.model_dirname)
    if os.path.exists(os.path.join(preferred, "config.json")):
        return preferred
    candidates = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
                candidates.append(path)
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        # several trained models: pick the most recently written one
        return max(candidates, key=lambda p: os.path.getmtime(os.path.join(p, "config.json")))
    return preferred


def write_profile_artifact(models_dir: str, profile: DatasetProfile) -> str:
    """Persist the profile beside the weights so demos use the right questions."""
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, "profile.json")
    payload: Dict[str, Any] = {
        "key": profile.key,
        "display_name": profile.display_name,
        "subject": profile.subject,
        "category_label": profile.category_label,
        "identity_label": profile.identity_label,
        "description": profile.description,
        "questions": profile.questions,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return path


def read_model_profile(models_dir: str) -> Dict[str, Any]:
    """Profile saved next to a trained model, or the air quality defaults."""
    path = os.path.join(models_dir, "profile.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {"key": AIR_QUALITY.key, "display_name": AIR_QUALITY.display_name,
            "subject": AIR_QUALITY.subject, "questions": AIR_QUALITY.questions,
            "category_label": AIR_QUALITY.category_label}
