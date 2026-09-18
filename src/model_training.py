"""
model_training.py

Trains, tunes, and evaluates multiple regression models for predicting
students' O-level Mathematics final_test score. At least 3 models are
compared so the best-performing one can be selected for identifying
weaker students prior to the exam.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .data_preparation import prepare_data


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------
def build_preprocessor(numerical_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Build a ColumnTransformer that scales numerical features and
    one-hot encodes categorical features. Wrapping this in each model's
    pipeline keeps preprocessing consistent across train/test/tuning.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),                                  # scale numerical features using standardization (mean=0, std=1)
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),          
        ]
    )
    return preprocessor


# ---------------------------------------------------------------------
# Models + hyperparameter grids
# ---------------------------------------------------------------------
def get_model_configs() -> dict:
    """
    Returns a dict of {model_name: (estimator, param_grid)}.

    Three models are evaluated:
      - Linear Regression: simple linear regression using a single predictor
        (the feature most correlated with final_test, set via
        SIMPLE_REGRESSION_FEATURE below).
      - Multivariate Regression: ordinary least squares using all available
        predictors (i.e. standard multiple linear regression).
      - Ridge Regression: L2-regularized multiple linear regression, tuned
        over alpha, using all available predictors.

    Param grids are prefixed with 'model__' to match the pipeline step name.
    """
    configs = {
        "Linear Regression": (
            LinearRegression(),
            {},  # no hyperparameters to tune
        ),
        "Multivariate Regression": (
            LinearRegression(),
            {},  # same estimator as above, fit on all predictors
        ),
        "Ridge Regression": (
            Ridge(random_state=42),
            {
                "model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0],      #different values of alpha to tune the Ridge regression model
            },
        ),
    }
    return configs


# Single predictor used for the "Linear Regression" (simple regression) model.
SIMPLE_REGRESSION_FEATURE = "attendance_rate"


# ---------------------------------------------------------------------
# Training + hyperparameter tuning
# ---------------------------------------------------------------------
def tune_model(
    estimator,
    param_grid: dict,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
):
    """
    Wrap the estimator in a Pipeline with preprocessing, then run
    GridSearchCV (or fit directly if no param grid is given).
    Returns the best fitted pipeline.
    """
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    if not param_grid:
        pipeline.fit(X_train, y_train)
        print("No hyperparameters to tune; fitted model directly.\n")
        return pipeline

    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"Best parameters found: {search.best_params_}")

    return search.best_estimator_


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute MSE, RMSE, MAE, and R^2 for a fitted model on the test set."""
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


# ---------------------------------------------------------------------
# Run + compare all models
# ---------------------------------------------------------------------
def run_experiments(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    numerical_features: list,
    categorical_features: list,
):
    """
    Tune and evaluate every model in get_model_configs().
    Returns:
        results_df   - comparison table of metrics per model
        fitted_models - dict of {model_name: fitted best pipeline}
    """
    configs = get_model_configs()

    results = []
    fitted_models = {}

    for name, (estimator, param_grid) in configs.items():
        print(f"Training and tuning: {name}...")

        if name == "Linear Regression":
            # Simple regression: single numerical predictor, no encoding needed
            preprocessor = build_preprocessor([SIMPLE_REGRESSION_FEATURE], [])
            X_train_subset = X_train[[SIMPLE_REGRESSION_FEATURE]]
            X_test_subset = X_test[[SIMPLE_REGRESSION_FEATURE]]
        else:
            # Multivariate / Ridge: all numerical + categorical predictors
            preprocessor = build_preprocessor(numerical_features, categorical_features)
            X_train_subset = X_train
            X_test_subset = X_test

        best_model = tune_model(estimator, param_grid, preprocessor, X_train_subset, y_train)
        metrics = evaluate_model(best_model, X_test_subset, y_test)
        metrics["Model"] = name
        results.append(metrics)
        fitted_models[name] = best_model

    results_df = pd.DataFrame(results)[["Model", "MSE", "RMSE", "MAE", "R2"]]
    results_df = results_df.sort_values("RMSE").reset_index(drop=True)

    return results_df, fitted_models


def select_best_model(results_df: pd.DataFrame, fitted_models: dict, metric: str = "RMSE"):
    """Return the (name, model) pair with the best score on the given metric."""
    ascending = metric != "R2"  # lower is better for RMSE/MAE, higher is better for R2
    best_name = results_df.sort_values(metric, ascending=ascending).iloc[0]["Model"]
    return best_name, fitted_models[best_name]