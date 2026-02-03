"""
Simple ML Pipeline with Hero ML Model Registry Integration

This script demonstrates how to:
1. Train multiple ML models with different hyperparameters
2. Log experiments, parameters, and metrics to Hero ML Model Registry
3. Save and track models for production use
"""

import argparse
from typing import Dict, List, Tuple

from dotenv import load_dotenv

import hero
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load environment variables from .env file
load_dotenv()


def load_data(test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, ...]:
    """Load and split the Iris dataset.

    Args:
        test_size: Proportion of dataset for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print("Loading Iris dataset...")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}\n")
    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of metric names and values
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
    }


def train_and_log_model(
    mlflow,
    experiment,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int,
    max_depth: int,
    run_name: str,
) -> float:
    """Train a Random Forest model and log to Hero ML Model Registry.

    Args:
        mlflow: Patched MLflow instance from Hero
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees
        run_name: Name for this MLflow run

    Returns:
        Test accuracy score
    """
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # Log hyperparameters
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": 42,
            "model_type": "RandomForestClassifier",
            "dataset": "iris",
        }

        for key, value in params.items():
            mlflow.log_param(key, value)

        # Train model
        print(f"Training {run_name}...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Perform cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
        mlflow.log_metric("cv_std_accuracy", cv_scores.std())

        # Calculate training metrics
        train_pred = model.predict(X_train)
        train_metrics = calculate_metrics(y_train, train_pred)
        for metric_name, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value)

        # Calculate test metrics
        test_pred = model.predict(X_test)
        test_metrics = calculate_metrics(y_test, test_pred)
        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        # Log feature importance
        feature_importance = dict(zip(
            [f"feature_{i}" for i in range(X_train.shape[1])],
            model.feature_importances_
        ))
        mlflow.log_params({f"importance_{k}": f"{v:.4f}" for k,
                          v in feature_importance.items()})

        # Save model
        mlflow.sklearn.log_model(model, "model")

        # Print results
        print(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Test F1 Score:  {test_metrics['f1_score']:.4f}")
        print(
            f"  CV Mean:        {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print()

        return test_metrics['accuracy']


def run_hyperparameter_sweep(
    mlflow,
    experiment,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    configs: List[Dict],
) -> None:
    """Run multiple training experiments with different hyperparameters.

    Args:
        mlflow: Patched MLflow instance from Hero
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        configs: List of hyperparameter configurations to try
    """
    print(
        f"Running hyperparameter sweep with {len(configs)} configurations...\n")
    print("=" * 60)

    results = []
    for config in configs:
        accuracy = train_and_log_model(
            mlflow,
            experiment,
            X_train,
            X_test,
            y_train,
            y_test,
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            run_name=config["run_name"],
        )
        results.append((config["run_name"], accuracy))

    print("=" * 60)
    print("\nSummary of Results:")
    print("-" * 60)
    for run_name, accuracy in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"  {run_name:30s} | Test Accuracy: {accuracy:.4f}")
    print("-" * 60)


def main():
    """Main function to run the ML pipeline."""
    parser = argparse.ArgumentParser(
        description="Train models with Hero ML Model Registry")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="Iris Classification Pipeline",
        help="Name of the MLflow experiment",
    )
    args = parser.parse_args()

    # Initialize Hero client and get patched MLflow
    print("Initializing Hero ML Model Registry...")
    hero_client = hero.HeroClient()
    model_registry = hero_client.MLModelRegistry()
    mlflow = model_registry.get_patched_mlflow()

    # Override tracking URI to use local database for development
    # mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_tracking_uri(model_registry.get_tracking_uri())

    tracking_uri = model_registry.get_tracking_uri()
    print(f"MLflow Tracking URI: {tracking_uri}\n")

    # Set experiment
    # Set experiment
    experiment_name = "rs_test_experiment_1"
    try:
        experiment = mlflow.set_experiment(experiment_name=experiment_name)
    except:
        experiment_id = mlflow.create_experiment(experiment_name)
        experiment = model_registry.read_experiment(experiment_id)
    print(f"Experiment: {experiment.name}")
    print(f"Experiment ID: {experiment.experiment_id}\n")

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Define hyperparameter configurations to test
    configs = [
        {"n_estimators": 50, "max_depth": 3, "run_name": "rf_50_trees_depth_3"},
        {"n_estimators": 100, "max_depth": 5, "run_name": "rf_100_trees_depth_5"},
        {"n_estimators": 150, "max_depth": 10,
            "run_name": "rf_150_trees_depth_10"},
        {"n_estimators": 200, "max_depth": 15,
            "run_name": "rf_200_trees_depth_15"},
        {"n_estimators": 100, "max_depth": None,
            "run_name": "rf_100_trees_no_limit"},
    ]

    # Run experiments
    run_hyperparameter_sweep(mlflow, experiment, X_train,
                             X_test, y_train, y_test, configs)

    # Show experiment info
    print(f"\nView your experiments at: {tracking_uri}")
    # print(
    #     f"All runs have been logged to Hero ML Model Registry under experiment '{args.experiment_name}'")


if __name__ == "__main__":
    main()
