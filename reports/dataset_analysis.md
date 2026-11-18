# Student Performance Dataset Analysis

Generated automatically by `src/train_slm.py`. Every number below was measured on the files found in the project at run time.

## 1. Data source

- Discovery method: **pre-existing split files**
- Searched: `C:\Users\gnaya\OneDrive\Desktop\design thinking\data\processed`
- Searched: `C:\Users\gnaya\OneDrive\Desktop\design thinking\data\training`
- Searched: `C:\Users\gnaya\OneDrive\Desktop\design thinking\data\validation`
- Searched: `C:\Users\gnaya\OneDrive\Desktop\design thinking\data\testing`
- Searched: `C:\Users\gnaya\OneDrive\Desktop\design thinking\data\test`
- Searched: `C:\Users\gnaya\OneDrive\Desktop\design thinking\data`
- Searched: `C:\Users\gnaya\OneDrive\Desktop\design thinking\hugging face`
- Searched: `C:\Users\gnaya\OneDrive\Desktop\design thinking`

## 2. Files examined

| file | rows | used | assessment |
|---|---|---|---|
| `test.jsonl` | 10,000 | no | no known profile matched |
| `test_qa.jsonl` | 4,000 | no | no known profile matched |
| `train.jsonl` | 100,008 | no | no known profile matched |
| `train_qa.jsonl` | 4,000 | no | no known profile matched |
| `validation.jsonl` | 10,000 | no | no known profile matched |
| `validation_qa.jsonl` | 4,000 | no | no known profile matched |
| `exams_test.parquet` | 150 | yes | student-performance table: demographics (`gender`); score (`math_score`) (gender (`gender`); lunch (`lunch`); parental education (`parental level of education`); scores (`math score`); scores (`reading score`); scores (`writing score`); test preparation (`test preparation course`)) |
| `exams_train.parquet` | 700 | yes | student-performance table: demographics (`gender`); score (`math_score`) (gender (`gender`); lunch (`lunch`); parental education (`parental level of education`); scores (`math score`); scores (`reading score`); scores (`writing score`); test preparation (`test preparation course`)) |
| `exams_validation.parquet` | 150 | yes | student-performance table: demographics (`gender`); score (`math_score`) (gender (`gender`); lunch (`lunch`); parental education (`parental level of education`); scores (`math score`); scores (`reading score`); scores (`writing score`); test preparation (`test preparation course`)) |
| `million_test.parquet` | 150,000 | yes | student-performance table: attendance (`attendance_percentage`); grade (`grade`); participation (`class_participation`); score (`total_score`); study behaviour (`weekly_self_study_hours`) (grades (`grade`); grades (`total_score`); study habits (`attendance_percentage`); study habits (`class_participation`); study habits (`weekly_self_study_hours`)) |
| `million_train_.parquet` | 700,000 | yes | student-performance table: attendance (`attendance_percentage`); grade (`grade`); participation (`class_participation`); score (`total_score`); study behaviour (`weekly_self_study_hours`) (grades (`grade`); grades (`total_score`); study habits (`attendance_percentage`); study habits (`class_participation`); study habits (`weekly_self_study_hours`)) |
| `million_validation.parquet` | 150,000 | yes | student-performance table: attendance (`attendance_percentage`); grade (`grade`); participation (`class_participation`); score (`total_score`); study behaviour (`weekly_self_study_hours`) (grades (`grade`); grades (`total_score`); study habits (`attendance_percentage`); study habits (`class_participation`); study habits (`weekly_self_study_hours`)) |

### Rejected files

- `C:\Users\gnaya\OneDrive\Desktop\design thinking\data\processed\student_performance\test.jsonl`: no known profile matched
- `C:\Users\gnaya\OneDrive\Desktop\design thinking\data\processed\student_performance\test_qa.jsonl`: no known profile matched
- `C:\Users\gnaya\OneDrive\Desktop\design thinking\data\processed\student_performance\train.jsonl`: no known profile matched
- `C:\Users\gnaya\OneDrive\Desktop\design thinking\data\processed\student_performance\train_qa.jsonl`: no known profile matched
- `C:\Users\gnaya\OneDrive\Desktop\design thinking\data\processed\student_performance\validation.jsonl`: no known profile matched
- `C:\Users\gnaya\OneDrive\Desktop\design thinking\data\processed\student_performance\validation_qa.jsonl`: no known profile matched

## 3. Split sizes

| split | rows | columns | duplicate rows | missing values |
|---|---|---|---|---|
| test | 2,000 | 6 | 0 | 0 |
| train | 20,000 | 6 | 0 | 0 |
| validation | 2,000 | 6 | 0 | 0 |

## 4. Records after preprocessing

| split | source rows | rows after cleaning | text documents | characters |
|---|---|---|---|---|
| test | 2,000 | 2,000 | 10,000 | 1,542,450 |
| train | 20,000 | 20,000 | 100,008 | 15,424,613 |
| validation | 2,000 | 2,000 | 10,000 | 1,542,486 |

Total serialised documents: **120,008** (18,509,549 characters).

## 5. Column analysis

### `test` split

| column | dtype | missing | range / values | mean / top | outliers (3·IQR) |
|---|---|---|---|---|---|
| `student_id` | int64 | 0 | 202.0000 .. 999856.0000 | 514370.5450 | 0 |
| `weekly_self_study_hours` | float64 | 0 | 0.0000 .. 38.0000 | 15.1517 | 0 |
| `attendance_percentage` | float64 | 0 | 52.3000 .. 100.0000 | 84.6886 | 0 |
| `class_participation` | float64 | 0 | 0.0000 .. 10.0000 | 5.9204 | 0 |
| `total_score` | float64 | 0 | 16.9000 .. 100.0000 | 84.7716 | 0 |
| `grade` | str | 0 | 5 distinct | A, B, C | - |

### `train` split

| column | dtype | missing | range / values | mean / top | outliers (3·IQR) |
|---|---|---|---|---|---|
| `student_id` | int64 | 0 | 41.0000 .. 999899.0000 | 496455.2612 | 0 |
| `weekly_self_study_hours` | float64 | 0 | 0.0000 .. 40.0000 | 15.0717 | 0 |
| `attendance_percentage` | float64 | 0 | 50.0000 .. 100.0000 | 84.6361 | 0 |
| `class_participation` | float64 | 0 | 0.0000 .. 10.0000 | 5.9945 | 0 |
| `total_score` | float64 | 0 | 18.1000 .. 100.0000 | 84.4552 | 0 |
| `grade` | str | 0 | 5 distinct | A, B, C | - |

### `validation` split

| column | dtype | missing | range / values | mean / top | outliers (3·IQR) |
|---|---|---|---|---|---|
| `student_id` | int64 | 0 | 897.0000 .. 999663.0000 | 498226.7520 | 0 |
| `weekly_self_study_hours` | float64 | 0 | 0.0000 .. 37.8000 | 15.0604 | 0 |
| `attendance_percentage` | float64 | 0 | 50.7000 .. 100.0000 | 84.6652 | 0 |
| `class_participation` | float64 | 0 | 0.0000 .. 10.0000 | 5.9977 | 0 |
| `total_score` | float64 | 0 | 24.5000 .. 100.0000 | 84.1694 | 0 |
| `grade` | str | 0 | 5 distinct | A, B, C | - |

## 6. Field classification

- **Category**: `grade`
- **Numeric**: `student_id`, `weekly_self_study_hours`, `attendance_percentage`, `class_participation`, `total_score`
- **Categorical**: `grade`
- **Identifier**: `student_id`

## 7. Split integrity

Leakage detected between splits: **False**

| split pair | shared rows | % of smaller split |
|---|---|---|
| test ↔ train | 0 | 0.0000% |
| test ↔ validation | 0 | 0.0000% |
| train ↔ validation | 0 | 0.0000% |

## 8. Preprocessing decisions

- Original files are opened **read-only** and never modified or deleted.
- Cleaning is **stateless and per-value** (whitespace stripping, numeric coercion, timestamp parsing, exact-duplicate removal within a split). No statistic is fitted on one split and applied to another, so no transformation can leak test information.
- Missing measurements are **not imputed**. A missing field is simply omitted from the serialised record instead of being filled with a guessed value.
- Outliers are **reported, not removed**, because extreme pollution episodes are the signal of interest in air quality data.
- `test`: stripped whitespace and empty strings in text columns (0 values normalised); dropped 0 exact duplicate rows within this split (duplicates are never removed across splits, and splits are never re-shuffled)
- `train`: stripped whitespace and empty strings in text columns (0 values normalised); dropped 0 exact duplicate rows within this split (duplicates are never removed across splits, and splits are never re-shuffled)
- `validation`: stripped whitespace and empty strings in text columns (0 values normalised); dropped 0 exact duplicate rows within this split (duplicates are never removed across splits, and splits are never re-shuffled)

## 9. Text serialisation example

```text
### Student record: for student 24261. Grade: B. Measurements: Weekly self study hours 8.6 hours per week; Attendance percentage 72.9 %; Class participation 4.9 points out of 10; Total score 79.8 points out of 100. ### End
```

```text
Question: What is the grade for student 24261?
Answer: The grade is B.
```

```text
Question: What is the weekly self study hours for student 24261?
Answer: The weekly self study hours is 8.6 hours per week.
```

```text
Question: What is the attendance percentage for student 24261?
Answer: The attendance percentage is 72.9 %.
```

```text
Question: What is the class participation for student 24261?
Answer: The class participation is 4.9 points out of 10.
```

