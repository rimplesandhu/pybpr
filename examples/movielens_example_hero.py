"""
MovieLens Example with Hero MLflow Integration

Uses Hero's ML Model Registry for experiment tracking and model logging.
Demonstrates data processing and training with Hero-backed MLflow.

Usage:
    python examples/movielens_example_hero.py \
        --config examples/training_config.yaml

SSL Note:
    If you encounter SSL certificate errors, set in .env:
    HERO_SSL_VERIFY=false
"""

import hero
import argparse
import os
import ssl
import urllib.request

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from pybpr import TrainingPipeline, UserItemData
from pybpr.movielens_loader import MovieLensDownloader

# Load environment variables for Hero authentication
load_dotenv()

# Handle SSL certificate verification for enterprise environments
# MUST be set BEFORE importing hero to affect JWT client
# if os.getenv('HERO_SSL_VERIFY', 'true').lower() == 'false':
# ssl._create_default_https_context = ssl._create_unverified_context
# # Patch urllib's ssl module
# urllib.request.ssl._create_default_https_context = (
#     ssl._create_unverified_context
# )
# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ['REQUESTS_CA_BUNDLE'] = ''
# os.environ['PYTHONHTTPSVERIFY'] = '0'

# Import hero AFTER SSL configuration


def initialize_hero_mlflow(experiment_name: str):
    """Initialize Hero client and configure MLflow."""
    print("Initializing Hero ML Model Registry...")
    hero_client = hero.HeroClient()
    model_registry = hero_client.MLModelRegistry()
    mlflow = model_registry.get_patched_mlflow()
    mlflow.set_tracking_uri(model_registry.get_tracking_uri())

    tracking_uri = model_registry.get_tracking_uri()
    print(f"MLflow Tracking URI: {tracking_uri}\n")

    # Read or create experiment
    experiment = model_registry.read_or_create_experiment(experiment_name)
    print(f"Experiment: {experiment.name}")
    print(f"Experiment ID: {experiment.experiment_id}\n")

    return mlflow, model_registry, experiment


def load_movielens_data(dataset: str = 'ml-100k', cache_dir: str = None):
    """Load and preprocess MovieLens data using downloader."""
    # Initialize downloader
    downloader = MovieLensDownloader(cache_dir=cache_dir)

    # Download and load dataset
    data = downloader.load_dataset_with_tags(dataset)

    # Get ratings DataFrame
    rdf = data['ratings'].copy()
    rdf.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

    # ml-100k uses movie genres as features
    movies = data['movies'].copy()
    # Extract genre columns (last 19 columns are genre indicators)
    genre_cols = [f'genre_{i}' for i in range(19)]
    genre_data = movies[['movieId'] + genre_cols].copy()

    # Create item features from genres
    item_features = []
    for _, row in genre_data.iterrows():
        movie_id = row['movieId']
        for i, genre_col in enumerate(genre_cols):
            if row[genre_col] == 1:
                item_features.append({'MovieID': movie_id, 'TagID': i})

    tdf = pd.DataFrame(item_features)

    # Keep only movies with features
    rdf = rdf[rdf.MovieID.isin(tdf.MovieID.unique())].copy()

    print(
        f"Loaded {len(rdf)} ratings for {rdf.UserID.nunique()} "
        f"users and {rdf.MovieID.nunique()} movies"
    )
    print(f"Loaded {len(tdf)} genre features")

    return rdf, tdf


def build_user_item_data(
    rdf: pd.DataFrame,
    tdf: pd.DataFrame,
    rating_threshold: float = 3.5,
    item_feature: str = 'metadata',
    neg_option: str = 'neg-ignore',
    name: str = 'movielens'
):
    """Build UserItemData from MovieLens dataframes."""
    ui = UserItemData(name=name)

    # Add positive interactions (high ratings)
    positive_mask = rdf.Rating >= rating_threshold
    ui.add_positive_interactions(
        user_ids=rdf.UserID[positive_mask],
        item_ids=rdf.MovieID[positive_mask]
    )

    # Add negative interactions (low ratings) if requested
    if neg_option != 'neg-ignore':
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
        default='training_config.yaml',
        help='Training config YAML path'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='ml-100k',
        choices=['ml-100k', 'ml-1m', 'ml-10m', 'ml-20m', 'ml-25m'],
        help='MovieLens dataset version'
    )
    parser.add_argument(
        '--cache-dir', type=str, default=None, help='Cache directory'
    )
    parser.add_argument(
        '--item-option',
        type=str,
        default='metadata',
        choices=['metadata', 'indicator', 'both'],
        help='Item feature type'
    )
    parser.add_argument(
        '--neg-option',
        type=str,
        default='neg-ignore',
        choices=['neg-ignore', 'neg-test', 'neg-both'],
        help='Negative interaction handling'
    )
    parser.add_argument(
        '--grid-search',
        action='store_true',
        help='Run grid search'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='[Demo] movielens_pybpr',
        help='MLflow experiment name'
    )

    args = parser.parse_args()

    # Initialize Hero MLflow
    mlflow, _, experiment = initialize_hero_mlflow(args.experiment_name)

    # Load data
    print("Loading MovieLens data...")
    rdf, tdf = load_movielens_data(
        dataset=args.dataset, cache_dir=args.cache_dir
    )

    # MovieLens-specific rating threshold for positive interactions
    rating_threshold = 3.5

    # Build UserItemData object
    print("\nBuilding UserItemData...")
    ui = build_user_item_data(
        rdf=rdf,
        tdf=tdf,
        rating_threshold=rating_threshold,
        item_feature=args.item_feature,
        neg_option=args.neg_option,
        name=f'{args.dataset}_{args.item_feature}_{args.neg_option}'
    )

    # Initialize training pipeline
    print(f"\nInitializing pipeline: {args.config}")
    pipeline = TrainingPipeline(config_path=args.config)

    # Define tags for tracking
    tags = {
        "Project": "pybpr MovieLens",
        "dataset": args.dataset,
        "item_feature": args.item_feature,
        "neg_option": args.neg_option
    }

    # Train model or run grid search
    if args.grid_search:
        print("\nRunning grid search...")
        # Define parameter grid
        param_grid = {
            'model.n_latent': [32, 64, 128],
            'optimizer.lr': [0.001, 0.01],
            'training.loss_function': ['bpr_loss', 'hinge_loss']
        }
        results = pipeline.run_grid_search(
            ui,
            param_grid=param_grid,
            mlflow_experiment_name=args.experiment_name,
            base_run_name=ui.name
        )
        print(f"\nGrid search completed: {len(results)} experiments")
    else:
        print("\nTraining single model...")
        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=ui.name
        ) as run:
            # Log tags
            mlflow.set_tags(tags)

            # Train model
            pipeline.train(ui, mlflow_run=run)

            print(f"\nTraining completed!")
            print(f"Experiment ID: {experiment.experiment_id}")
            print(f"Run ID: {run.info.run_id}")


if __name__ == '__main__':
    main()
