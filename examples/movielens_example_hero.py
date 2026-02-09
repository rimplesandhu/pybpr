"""
MovieLens Example with Hero MLflow Integration

Uses Hero's ML Model Registry for experiment tracking and model logging.
Demonstrates data processing and training with Hero-backed MLflow using TrainingPipeline.

Usage:
    python examples/movielens_example_hero.py \
        --config examples/config.yaml [--sweep]
"""

import argparse

import hero
import numpy as np
from dotenv import load_dotenv

from pybpr import TrainingPipeline, UserItemData, load_movielens

# Load environment variables for Hero authentication
load_dotenv()


def initialize_hero_mlflow():
    """Initialize Hero client and configure MLflow."""
    print("Initializing Hero ML Model Registry...")
    hero_client = hero.HeroClient()
    model_registry = hero_client.MLModelRegistry()
    mlflow = model_registry.get_patched_mlflow()
    mlflow.set_tracking_uri(model_registry.get_tracking_uri())

    tracking_uri = model_registry.get_tracking_uri()
    print(f"MLflow Tracking URI: {tracking_uri}\n")

    return mlflow


def build_user_item_data(
    data: dict,
    rating_threshold: float,
    item_feature: str,
    name: str
):
    """Build UserItemData from MovieLens data dict."""
    # Extract ratings and features from data dict
    rdf = data['ratings']
    tdf = data['features']

    # Initialize UserItemData
    ui = UserItemData(name=name)

    # Add positive interactions (high ratings)
    positive_mask = rdf.Rating >= rating_threshold
    ui.add_positive_interactions(
        user_ids=rdf.UserID[positive_mask],
        item_ids=rdf.MovieID[positive_mask]
    )

    # Add negative interactions (low ratings)
    negative_mask = rdf.Rating < rating_threshold
    ui.add_negative_interactions(
        user_ids=rdf.UserID[negative_mask],
        item_ids=rdf.MovieID[negative_mask]
    )

    # Add user features (identity mapping)
    ui.add_user_features(
        user_ids=rdf.UserID.unique(),
        feature_ids=rdf.UserID.unique()
    )

    # Add item features based on option
    if item_feature == 'metadata':
        # Use only tag features
        ui.add_item_features(
            item_ids=tdf.MovieID,
            feature_ids=tdf.TagID
        )
    elif item_feature == 'indicator':
        # Use only indicator features (one-hot)
        ui.add_item_features(
            item_ids=tdf.MovieID.unique(),
            feature_ids=tdf.MovieID.unique()
        )
    elif item_feature == 'both':
        # Combine tag features and indicator features
        ui.add_item_features(
            item_ids=np.concatenate(
                [tdf.MovieID.values, tdf.MovieID.unique()]
            ),
            feature_ids=np.concatenate([
                tdf.TagID.values,
                tdf.TagID.max() + tdf.MovieID.unique()
            ])
        )
    else:
        raise ValueError(f"Unknown item_feature: {item_feature}")

    print(ui)
    return ui


def main():
    parser = argparse.ArgumentParser(
        description='Train MovieLens with Hero MLflow'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Training config YAML path'
    )
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='Run parameter sweep instead of single training'
    )

    args = parser.parse_args()

    # Initialize Hero MLflow
    hero_mlflow = initialize_hero_mlflow()

    # Initialize training pipeline
    print(f"Loading config: {args.config}")
    pipeline = TrainingPipeline(config_path=args.config)

    # Load data using simple function
    print("Loading MovieLens 100K data...")
    data = load_movielens(dataset='ml-100k', preprocess=True)

    # MovieLens-specific rating threshold for positive interactions
    rating_threshold = 3.5

    # Build UserItemData object
    print("\nBuilding UserItemData...")
    ui = build_user_item_data(
        data=data,
        rating_threshold=rating_threshold,
        item_feature=pipeline.cfg.get('data.item_feature'),
        name=f'ml-100k_{pipeline.cfg.get("data.item_feature")}'
    )

    # Run training or sweep
    pipeline.run(ui, sweep=args.sweep, custom_mlflow=hero_mlflow)


if __name__ == '__main__':
    main()
