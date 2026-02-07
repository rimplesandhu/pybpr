"""MLflow experiment visualization utilities."""

from pathlib import Path
from typing import Optional, List, Dict, Union

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


class MLflowPlotter:
    """Plot metrics from MLflow experiments."""

    def __init__(self, tracking_uri: Union[str, Path] = "mlflow.db"):
        """Initialize plotter with MLflow tracking URI.

        Args:
            tracking_uri: Path to MLflow database or tracking URI
        """
        # Convert path to SQLite URI if needed
        if not str(tracking_uri).startswith(("sqlite://", "http")):
            tracking_uri = f"sqlite:///{tracking_uri}"

        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.MlflowClient(tracking_uri=tracking_uri)

    def get_experiments(self) -> pd.DataFrame:
        """Get all experiments as DataFrame."""
        exps = self.client.search_experiments()
        return pd.DataFrame([
            {
                "experiment_id": e.experiment_id,
                "name": e.name,
                "artifact_location": e.artifact_location,
            }
            for e in exps
        ])

    def get_runs(
        self,
        experiment_name: Optional[str] = None,
        experiment_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Get runs for an experiment.

        Args:
            experiment_name: Name of experiment
            experiment_id: ID of experiment (overrides name)

        Returns:
            DataFrame with run information
        """
        # Get experiment ID from name if needed
        if experiment_id is None and experiment_name is not None:
            exp = self.client.get_experiment_by_name(experiment_name)
            if exp is None:
                raise ValueError(f"Experiment '{experiment_name}' not found")
            experiment_id = exp.experiment_id

        # Search runs
        runs = self.client.search_runs(
            experiment_ids=[experiment_id] if experiment_id else None
        )

        # Build DataFrame
        data = []
        for run in runs:
            row = {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
            }
            row.update(run.data.params)
            row.update(run.data.metrics)
            data.append(row)

        return pd.DataFrame(data)

    def get_run_metrics_history(
        self, run_id: str, metric_keys: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Get metric history for a run.

        Args:
            run_id: MLflow run ID
            metric_keys: List of metric names to fetch

        Returns:
            Dict mapping metric name to DataFrame with columns:
            [step, value, timestamp]
        """
        # Get all metric keys if not specified
        if metric_keys is None:
            run = self.client.get_run(run_id)
            metric_keys = list(run.data.metrics.keys())

        # Fetch history for each metric
        histories = {}
        for key in metric_keys:
            history = self.client.get_metric_history(run_id, key)
            histories[key] = pd.DataFrame([
                {
                    "step": m.step,
                    "value": m.value,
                    "timestamp": m.timestamp,
                }
                for m in history
            ])

        return histories

    def plot_runs_comparison(
        self,
        experiment_name: str,
        metrics: List[str] = ["train_loss", "test_auc"],
        figsize: tuple = (14, 5),
        std_width: float = 1.0,
        show_std: bool = True
    ) -> Figure:
        """Plot loss and AUC side by side with optional std bands.

        Args:
            experiment_name: Name of experiment
            metrics: List of metrics to plot (expects loss and auc)
            figsize: Figure size
            std_width: Std deviation multiplier for bands (1.0, 2.0, etc)
            show_std: Whether to show std bands on AUC plot

        Returns:
            Matplotlib Figure
        """
        # Get experiment
        exp = self.client.get_experiment_by_name(experiment_name)
        if exp is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        # Get all runs
        runs = self.client.search_runs(experiment_ids=[exp.experiment_id])

        if len(runs) == 0:
            raise ValueError(
                f"No runs found in experiment '{experiment_name}'"
            )

        # Create two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Collect data for std calculation
        metric_data = {m: {} for m in metrics}

        # Plot each metric
        for metric_idx, metric_name in enumerate(metrics):
            ax = axes[metric_idx]

            # Plot each run
            for run in runs:
                try:
                    history = self.client.get_metric_history(
                        run.info.run_id, metric_name
                    )
                    if len(history) == 0:
                        continue

                    steps = [m.step for m in history]
                    values = [m.value for m in history]

                    run_name = (
                        run.info.run_name or run.info.run_id[:8]
                    )
                    ax.plot(
                        steps, values, label=run_name, marker='o',
                        markersize=3, alpha=0.7
                    )

                    # Store for std calculation
                    for s, v in zip(steps, values):
                        if s not in metric_data[metric_name]:
                            metric_data[metric_name][s] = []
                        metric_data[metric_name][s].append(v)

                except Exception:
                    continue

            # Add std bands for AUC metrics
            if (show_std and 'auc' in metric_name.lower() and
                    metric_data[metric_name]):
                steps = sorted(metric_data[metric_name].keys())
                means = [
                    np.mean(metric_data[metric_name][s]) for s in steps
                ]
                stds = [
                    np.std(metric_data[metric_name][s]) for s in steps
                ]
                upper = [
                    m + std_width * s for m, s in zip(means, stds)
                ]
                lower = [
                    m - std_width * s for m, s in zip(means, stds)
                ]

                ax.fill_between(
                    steps, lower, upper, alpha=0.2, color='gray',
                    label=f'±{std_width}σ'
                )

            ax.set_xlabel("Step/Epoch")
            ax.set_ylabel(metric_name.replace("_", " ").title())
            ax.set_title(f"{metric_name.replace('_', ' ').title()}")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Experiment: {experiment_name}", fontsize=14, y=1.02
        )
        fig.tight_layout()
        return fig

    def plot_single_run(
        self,
        run_id: str,
        figsize: tuple = (14, 5),
        std_width: float = 2.0,
        show_std: bool = True
    ) -> Figure:
        """Plot train/test loss and AUC with std bands.

        Args:
            run_id: MLflow run ID
            figsize: Figure size
            std_width: Std multiplier for bands (default 2.0)
            show_std: Show ± std bands for AUC (default True)

        Returns:
            Matplotlib Figure
        """
        # Get run and all available metrics
        run = self.client.get_run(run_id)
        all_metric_keys = list(run.data.metrics.keys())

        if len(all_metric_keys) == 0:
            raise ValueError(f"No metrics found for run {run_id}")

        # Get metric histories for all metrics
        histories = self.get_run_metrics_history(
            run_id, all_metric_keys
        )

        # Group metrics by type (loss vs auc, excluding std)
        loss_metrics = [
            k for k in histories.keys()
            if 'loss' in k.lower()
        ]
        auc_metrics = [
            k for k in histories.keys()
            if 'auc' in k.lower() and '_std' not in k.lower()
        ]

        # Create two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot loss metrics on left subplot
        ax_loss = axes[0]
        for metric_name in loss_metrics:
            df = histories[metric_name]
            if len(df) > 0:
                ax_loss.plot(
                    df["step"], df["value"], marker='o',
                    markersize=4, linewidth=2, label=metric_name
                )
        ax_loss.set_xlabel("Step/Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Loss")
        ax_loss.legend(loc="best")
        ax_loss.grid(True, alpha=0.3)

        # Plot AUC metrics on right subplot with std bands
        ax_auc = axes[1]

        # Plot each AUC metric with its std band
        for idx, metric_name in enumerate(auc_metrics):
            df = histories[metric_name]
            if len(df) == 0:
                continue

            # Get corresponding std metric if available
            std_key = f"{metric_name}_std"
            df_std = histories.get(std_key)

            # Plot mean line
            color = f'C{idx}'
            ax_auc.plot(
                df["step"], df["value"], marker='o',
                markersize=4, linewidth=2,
                label=metric_name, color=color
            )

            # Add std bands if available and requested
            if show_std and df_std is not None and len(df_std) > 0:
                steps = df["step"].values
                means = df["value"].values
                stds = df_std["value"].values
                upper = means + std_width * stds
                lower = means - std_width * stds

                ax_auc.fill_between(
                    steps, lower, upper, alpha=0.2,
                    color=color, label=f'{metric_name} ±{std_width}σ'
                )

        ax_auc.set_xlabel("Step/Epoch")
        ax_auc.set_ylabel("AUC")
        ax_auc.set_title("AUC")
        ax_auc.legend(loc="best")
        ax_auc.grid(True, alpha=0.3)

        run_name = run.info.run_name or run.info.run_id[:8]
        fig.suptitle(f"Run: {run_name}", fontsize=14, y=1.02)
        fig.tight_layout()
        return fig

    def plot_best_runs(
        self,
        experiment_name: str,
        metric: str = "test_auc",
        n_best: int = 5,
        plot_metrics: List[str] = ["train_loss", "test_auc"],
        figsize: tuple = (14, 5),
        std_width: float = 1.0,
        show_std: bool = True
    ) -> Figure:
        """Plot top N runs with loss and AUC side by side.

        Args:
            experiment_name: Name of experiment
            metric: Metric to rank by (higher is better)
            n_best: Number of top runs to plot
            plot_metrics: List of metrics to plot
            figsize: Figure size
            std_width: Std deviation multiplier for bands
            show_std: Whether to show std bands on AUC plot

        Returns:
            Matplotlib Figure
        """
        # Get experiment
        exp = self.client.get_experiment_by_name(experiment_name)
        if exp is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        # Search runs sorted by metric
        runs = self.client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=n_best
        )

        if len(runs) == 0:
            raise ValueError(
                f"No runs found in experiment '{experiment_name}'"
            )

        # Create two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Collect data for std calculation
        metric_data = {m: {} for m in plot_metrics}

        # Plot each metric
        for metric_idx, metric_name in enumerate(plot_metrics):
            ax = axes[metric_idx]

            # Plot each run
            for rank, run in enumerate(runs, 1):
                try:
                    history = self.client.get_metric_history(
                        run.info.run_id, metric_name
                    )
                    if len(history) == 0:
                        continue

                    steps = [m.step for m in history]
                    values = [m.value for m in history]

                    run_name = (
                        run.info.run_name or run.info.run_id[:8]
                    )
                    final_val = run.data.metrics.get(metric, 0)
                    label = (
                        f"#{rank} {run_name} "
                        f"({metric}={final_val:.4f})"
                    )

                    ax.plot(
                        steps, values, label=label, marker='o',
                        markersize=3, alpha=0.7, linewidth=2
                    )

                    # Store for std calculation
                    for s, v in zip(steps, values):
                        if s not in metric_data[metric_name]:
                            metric_data[metric_name][s] = []
                        metric_data[metric_name][s].append(v)

                except Exception:
                    continue

            # Add std bands for AUC metrics
            if (show_std and 'auc' in metric_name.lower() and
                    metric_data[metric_name]):
                steps = sorted(metric_data[metric_name].keys())
                means = [
                    np.mean(metric_data[metric_name][s]) for s in steps
                ]
                stds = [
                    np.std(metric_data[metric_name][s]) for s in steps
                ]
                upper = [
                    m + std_width * s for m, s in zip(means, stds)
                ]
                lower = [
                    m - std_width * s for m, s in zip(means, stds)
                ]

                ax.fill_between(
                    steps, lower, upper, alpha=0.2, color='gray',
                    label=f'±{std_width}σ'
                )

            ax.set_xlabel("Step/Epoch")
            ax.set_ylabel(metric_name.replace("_", " ").title())
            ax.set_title(f"{metric_name.replace('_', ' ').title()}")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Top {n_best} Runs by {metric} - {experiment_name}",
            fontsize=14,
            y=1.02
        )
        fig.tight_layout()
        return fig

    def summary_table(
        self,
        experiment_name: str,
        metrics: List[str] = ["test_auc", "train_loss"],
        params: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Create summary table of runs.

        Args:
            experiment_name: Name of experiment
            metrics: Metrics to include (final values)
            params: Parameters to include

        Returns:
            DataFrame with run summaries
        """
        # Get runs
        runs_df = self.get_runs(experiment_name=experiment_name)

        # Select columns
        cols = ["run_name", "status"]
        if params:
            cols.extend([p for p in params if p in runs_df.columns])
        cols.extend([m for m in metrics if m in runs_df.columns])

        return runs_df[cols].sort_values(
            by=metrics[0] if metrics else "run_name", ascending=False
        )
