"""
main.py

Entry point for the O-Level Mathematics score prediction pipeline.
Loads config, prepares data, trains/tunes/evaluates the models defined
in src/model_training.py, and reports the results.
"""
import yaml
import os
from src.data_preparation import prepare_data
from src.model_training import run_experiments, select_best_model


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        # Get the directory of main.py and point to the config.yaml inside src/
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "src", "config.yaml")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config



def main():
    config = load_config()

#    data_path = config["data_path"]
    data_path = "/home/samuel/aiap/AIML_foundation/AI-ML-foundation-regression-/data/regression_bonus_practice_data.csv"
    numerical_features = config["numerical_features"]
    categorical_features = config["categorical_features"]
    target = config.get("target", "final_test")
    test_size = config.get("test_size", 0.2)
    random_state = config.get("random_state", 42)

    X_train, X_test, y_train, y_test = prepare_data(
        data_path, target=target, test_size=test_size, random_state=random_state
    )

    results_df, fitted_models = run_experiments(
        X_train, X_test, y_train, y_test,
        numerical_features, categorical_features,
    )

    print("\nModel comparison:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()