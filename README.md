# O-Level Mathematics Score Prediction

## Objective

Predict students' O-level Mathematics final examination score (`final_test`) using historical student data, so the school can identify weaker students **prior to the examination** and provide timely support/intervention.

This is framed as a regression problem: given student attributes (demographics, habits, attendance, family background, etc.), estimate their likely final test score.

## Dataset

- **Source file:** `data/regression_bonus_practice_data.csv`
- **Rows:** 15,900 students
- **Target variable:** `final_test` (numeric score)

**Features:**

| Column | Type | Description |
|---|---|---|
| `number_of_siblings` | numerical | Number of siblings |
| `n_male` | numerical | Number of male students in class |
| `n_female` | numerical | Number of female students in class |
| `age` | numerical | Student age |
| `hours_per_week` | numerical | Study hours per week |
| `attendance_rate` | numerical | Attendance rate (%) |
| `direct_admission` | categorical | Whether admitted directly |
| `CCA` | categorical | Co-curricular activity |
| `learning_style` | categorical | Preferred learning style |
| `gender` | categorical | Student gender |
| `tuition` | categorical | Whether student has tuition (Yes/No) |
| `sleep_time` | categorical | Usual sleep time |
| `wake_time` | categorical | Usual wake time |
| `mode_of_transport` | categorical | Mode of transport to school |
| `student_id`, `bag_color` | — | Dropped: identifiers/non-predictive columns |

Exploratory Data Analysis (distributions, missing values, outliers, and each feature's relationship to `final_test`) is documented in **`eda.ipynb`**.

## Project Structure

```
root/
├── eda.ipynb              # Exploratory data analysis & cleaning prototyping
├── README.md              
├── requirements.txt        # Python dependencies
├── data/
│   └── data.csv             # Raw dataset
├── src/
│   ├── data_preparation.py # Data loading, cleaning, train/test split
│   ├── model_training.py   # Model training, hyperparameter tuning, evaluation
│   └── config.yaml         # Data path, feature lists, split settings
└── main.py                 # Pipeline entry point
```

## Data Preparation (`src/data_preparation.py`)

Cleaning steps applied (ported from `eda.ipynb`):

1. **Missing values**
   - `CCA`: missing values filled with `"NONE"` (assumed no CCA participation)
   - `final_test`: rows with a missing target are dropped
   - `attendance_rate`: missing values filled with the median
2. **Invalid values**
   - Negative or implausible ages (`< 15`) imputed with the median age
3. **Standardization**
   - `tuition`: `"Y"`/`"N"` values standardized to `"Yes"`/`"No"`
   - All categorical text columns: whitespace stripped, title-cased
4. **Column dropping**
   - Non-predictive identifier columns (`student_id`, `bag_color`) removed
5. **Train/test split**
   - 80/20 split (configurable via `config.yaml`), `random_state=42` for reproducibility

## Models Evaluated (`src/model_training.py`)

Per the course syllabus (Chapter 4: Model Development), three regression models are trained, tuned, and compared:

| Model | Predictors used | Hyperparameter tuning |
|---|---|---|
| **Simple Linear Regression** | Single feature (`attendance_rate`, most correlated with `final_test`) | None (no hyperparameters) |
| **Multivariate (Multiple) Linear Regression** | All numerical + categorical features | None (no hyperparameters) |
| **Ridge Regression** | All numerical + categorical features | `alpha` tuned via `GridSearchCV` (5-fold CV) |

**Preprocessing:** numerical features are standardized (`StandardScaler`); categorical features are one-hot encoded (`OneHotEncoder`). Both are applied inside a `scikit-learn` `Pipeline` per model to prevent data leakage between train and test sets.

**Evaluation metrics:**

- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

The model with the lowest RMSE on the held-out test set is selected as the best model.

## Setup

1. Clone the repository and navigate to the project root.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure the dataset is placed at `data/data.csv` (or update `data_path` in `src/config.yaml` to point to your file).

## Usage

Run the full pipeline (data preparation → model training → tuning → evaluation) from the project root:

```bash
python main.py
```

This will:
1. Load and clean the dataset
2. Split it into train/test sets
3. Train and tune all three models
4. Print a comparison table of MSE, RMSE, MAE, and R² for each model