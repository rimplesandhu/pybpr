"""
MovieLens Example using TrainingPipeline

Demonstrates data processing and training separation:
1. Data loading creates UserItemData object
2. TrainingPipeline handles training with MLflow tracking

Usage:
    python examples/movielens_example.py \
        --config examples/config.yaml [--sweep]
"""

import argparse

import numpy as np

from pybpr import TrainingPipeline, UserItemData, load_movielens


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
        description='Train MovieLens using TrainingPipeline'
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

    # Initialize training pipeline
    print(f"Loading config: {args.config}")
    pipeline = TrainingPipeline(config_path=args.config)

    # Load data using simple function
    print("Loading MovieLens 100K data...")
    data = load_movielens(dataset='ml-100k', preprocess=True)

    # MovieLens-specific rating threshold for positive interactions
    rating_threshold = 3.0

    # Build UserItemData object
    print("\nBuilding UserItemData...")
    ui = build_user_item_data(
        data=data,
        rating_threshold=rating_threshold,
        item_feature=pipeline.cfg.get('data.item_feature'),
        name=f'ml-100k_{pipeline.cfg.get("data.item_feature")}'
    )

    # Run training or sweep
    run_ids = pipeline.run(ui, sweep=args.sweep)

    # Plot single run and save to figs/
    if not args.sweep and run_ids:
        import os
        import matplotlib.pyplot as plt
        from pybpr.plotter import MLflowPlotter

        # Create figs directory if it doesn't exist
        os.makedirs('figs', exist_ok=True)

        # Initialize plotter
        plotter = MLflowPlotter(
            tracking_uri=pipeline.cfg['mlflow.tracking_uri']
        )

        # Plot the single run
        fig = plotter.plot_single_run(
            run_id=run_ids[0],
            figsize=(14, 5),
            std_width=2.0,
            show_std=True
        )

        # Save the plot with run_id in filename
        save_path = f'figs/single_run_{run_ids[0]}.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
        plt.close(fig)


if __name__ == '__main__':
    main()
