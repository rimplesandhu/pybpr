"""Training pipeline for PyBPR recommendation system."""

import yaml
import torch
import itertools
import logging
import multiprocessing as mp
from functools import partial
from typing import Dict, List, Any, Optional, Callable

import mlflow
from mlflow.entities import Run

from .recommender import RecommendationSystem
from .interaction_data import UserItemData
from .matrix_factorization import HybridMF
from .losses import bpr_loss, hinge_loss, bpr_loss_v2, warp_loss

# Suppress verbose MLflow/alembic migration logs
logging.getLogger("alembic").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)


class TrainingPipeline:
    """Generic training pipeline for recommendation systems."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """Initialize pipeline with config from path or dict."""
        # Load or set configuration
        if config_path is not None:
            raw_config = self._load_config(config_path)
        elif config is not None:
            raw_config = config
        else:
            raise ValueError(
                "Must provide either config_path or config dict"
            )

        # Store raw and flattened versions
        self.cfg_raw = raw_config
        self.cfg = self._flatten_config(raw_config)

        # Map loss function names to actual functions
        self.loss_function_map = {
            'bpr_loss': bpr_loss,
            'bpr_loss_v2': bpr_loss_v2,
            'hinge_loss': hinge_loss,
            'warp_loss': warp_loss,
        }

    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    @staticmethod
    def _flatten_config(config: Dict) -> Dict:
        """Flatten nested config dict to single level with dots."""
        flat = {}
        for section, values in config.items():
            if isinstance(values, dict):
                for key, val in values.items():
                    flat[f"{section}.{key}"] = val
            else:
                flat[section] = values
        return flat

    def get_loss_function(self, loss_name: str) -> Callable:
        """Get loss function by name."""
        if loss_name not in self.loss_function_map:
            available = ', '.join(self.loss_function_map.keys())
            raise ValueError(
                f"Unknown loss function: {loss_name}. "
                f"Available options: {available}"
            )
        return self.loss_function_map[loss_name]

    def get_optimizer(
        self, optimizer_name: str, **kwargs
    ) -> partial:
        """Get optimizer by name with parameters."""
        # Map optimizer names to torch classes
        optimizer_map = {
            'Adam': torch.optim.Adam,
            'SGD': torch.optim.SGD,
            'AdamW': torch.optim.AdamW,
            'RMSprop': torch.optim.RMSprop,
        }

        # Validate optimizer name
        if optimizer_name not in optimizer_map:
            available = ', '.join(optimizer_map.keys())
            raise ValueError(
                f"Unknown optimizer: {optimizer_name}. "
                f"Available options: {available}"
            )

        return partial(optimizer_map[optimizer_name], **kwargs)

    def build_model(self, ui: UserItemData) -> HybridMF:
        """Build a HybridMF model from configuration."""
        # Build model from flattened config
        return HybridMF(
            n_user_features=ui.n_user_features,
            n_item_features=ui.n_item_features,
            n_latent=self.cfg['model.n_latent'],
            use_user_bias=self.cfg['model.use_user_bias'],
            use_global_bias=self.cfg['model.use_global_bias'],
            dropout=self.cfg['model.dropout'],
            activation=self.cfg['model.activation'],
            sparse=self.cfg['model.sparse']
        )

    def run(
        self, ui: UserItemData, sweep: bool = False
    ) -> List[str]:
        """Run training with optional sweep."""
        # Set MLflow tracking and experiment
        mlflow.set_tracking_uri(self.cfg['mlflow.tracking_uri'])
        mlflow.set_experiment(
            self.cfg['mlflow.experiment_name']
        )

        # Run sweep or single training
        if sweep:
            sweep_config = self.cfg_raw.get('sweep', {})
            if not sweep_config:
                raise ValueError("sweep config is empty")

            print("\nRunning parameter sweep...")
            results = self.run_grid_search(
                ui,
                param_grid=sweep_config,
                mlflow_experiment_name=self.cfg.get(
                    'mlflow.experiment_name'
                ),
                base_run_name=ui.name
            )
            print(
                f"\nSweep completed: {len(results)} experiments"
            )
            return results
        else:
            print("\nTraining single model...")
            with mlflow.start_run(run_name=ui.name) as run:
                self.train(ui, mlflow_run=run)
                print("\nTraining completed!")
                print(f"MLflow run ID: {run.info.run_id}")
                return [run.info.run_id]

    def train(
        self,
        ui: UserItemData,
        mlflow_run: Run,
        run_name: Optional[str] = None
    ) -> RecommendationSystem:
        """Train a single model using pipeline config."""
        # Determine run name
        if run_name is None:
            run_name = ui.name

        # Log all config parameters to MLflow
        mlflow.log_params(self.cfg)

        # Get loss function
        loss_name = self.cfg['training.loss_function']
        loss_fn = self.get_loss_function(loss_name)

        # Build model
        model = self.build_model(ui)

        # Build optimizer
        optimizer_name = self.cfg['optimizer.name']
        optimizer_params = {
            k.split('.')[1]: v
            for k, v in self.cfg.items()
            if (k.startswith('optimizer.')
                and k != 'optimizer.name')
        }
        optimizer = self.get_optimizer(
            optimizer_name, **optimizer_params
        )

        # Build RecommendationSystem with raw matrices
        print(f"Starting training: {run_name}", flush=True)
        device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'cpu'
        )
        train_ratio = self.cfg.get(
            'data.train_ratio_pos', 0.8
        )
        recommender = RecommendationSystem(
            Rpos=ui.Rpos,
            Rneg=ui.Rneg,
            Fu=ui.Fu,
            Fi=ui.Fi,
            model=model,
            optimizer=optimizer,
            loss_function=loss_fn,
            device=device,
            mlflow_run=mlflow_run,
            train_ratio=train_ratio,
        )

        # Train the model
        recommender.fit(
            n_iter=self.cfg['training.n_iter'],
            batch_size=self.cfg['training.batch_size'],
            eval_every=self.cfg['training.eval_every'],
            eval_user_size=self.cfg['training.eval_user_size'],
            early_stopping_patience=self.cfg[
                'training.early_stopping_patience'
            ]
        )

        print(f"Finished training: {run_name}", flush=True)
        return recommender

    def run_grid_search(
        self,
        ui: UserItemData,
        param_grid: Dict[str, List],
        mlflow_experiment_name: Optional[str] = None,
        base_run_name: Optional[str] = None,
        num_processes: Optional[int] = None
    ) -> List[str]:
        """Run hyperparameter grid search with param combinations."""
        # Validate param grid
        if not param_grid:
            raise ValueError(
                "param_grid cannot be empty. Provide parameter "
                "combinations for grid search."
            )

        # Set MLflow experiment
        if mlflow_experiment_name:
            mlflow.set_experiment(mlflow_experiment_name)

        # Generate all parameter combinations
        all_params = self._generate_param_combinations(param_grid)
        print(f"Running {len(all_params)} experiments in grid search")

        # Auto-configure multiprocessing for optimal performance
        total_cores = mp.cpu_count()

        # Determine number of parallel processes
        if num_processes is None:
            # Auto: use all cores, limited by number of experiments
            num_processes = min(len(all_params), total_cores)
        else:
            num_processes = min(len(all_params), num_processes)

        # Auto-calculate PyTorch threads to avoid oversubscription
        torch_num_threads = max(1, total_cores // num_processes)

        # Set PyTorch threads
        torch.set_num_threads(torch_num_threads)

        print(
            f"Using {num_processes} processes × "
            f"{torch_num_threads} PyTorch threads "
            f"({total_cores} cores available)"
        )

        # Create partial function with fixed ui
        run_single = partial(
            self._run_single_experiment,
            ui=ui,
            base_run_name=base_run_name or ui.name
        )

        # Run experiments in parallel with progress tracking
        print("Starting experiments...", flush=True)
        with mp.Pool(processes=num_processes) as pool:
            # Use imap_unordered for progress tracking
            results = []
            for i, result in enumerate(
                pool.imap_unordered(run_single, all_params), 1
            ):
                results.append(result)
                print(
                    f"[{i}/{len(all_params)}] {result}",
                    flush=True
                )

        # Print summary
        print("\nGrid Search Complete!")
        success_count = sum(
            1 for r in results if r.startswith("SUCCESS")
        )
        print(
            f"Success: {success_count}/{len(results)} experiments"
        )

        return results

    def _generate_param_combinations(
        self,
        param_grid: Dict[str, List]
    ) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters from grid."""
        # Extract parameter names and values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        # Generate all combinations
        all_combinations = []
        for values in itertools.product(*param_values):
            params = dict(zip(param_names, values))
            all_combinations.append(params)

        return all_combinations

    def _run_single_experiment(
        self,
        params: Dict[str, Any],
        ui: UserItemData,
        base_run_name: str
    ) -> str:
        """Run a single experiment with given parameters."""
        run_name = "unknown"
        try:
            # Create copy of config for this experiment
            experiment_config = self.cfg_raw.copy()

            # Generate run name from parameters
            run_name_parts = [base_run_name]

            # Update experiment config with param grid values
            for param_name, param_value in params.items():
                if '.' in param_name:
                    section, key = param_name.split('.', 1)
                    if section not in experiment_config:
                        experiment_config[section] = {}
                    experiment_config[section][key] = param_value

                    # Add to run name
                    if key == 'loss_function':
                        if callable(param_value):
                            run_name_parts.append(
                                param_value.__name__
                            )
                        else:
                            run_name_parts.append(str(param_value))
                    else:
                        run_name_parts.append(f"{key}{param_value}")

            run_name = '_'.join(str(p) for p in run_name_parts)

            # Create new pipeline with experiment config
            experiment_pipeline = TrainingPipeline(
                config=experiment_config
            )

            # Start MLflow run for this experiment
            with mlflow.start_run(run_name=run_name) as run:
                # Train model with MLflow run
                experiment_pipeline.train(
                    ui=ui, mlflow_run=run, run_name=run_name
                )

            return f"SUCCESS: {run_name}"

        except Exception as e:
            import traceback
            error_msg = (
                f"FAILED: {run_name}\n"
                f"  Params: {params}\n"
                f"  Error: {str(e)}\n"
                f"  {traceback.format_exc()}"
            )
            return error_msg

    @staticmethod
    def create_default_config() -> Dict:
        """Create a default configuration dictionary."""
        return {
            'model': {
                'n_latent': 64,
                'use_user_bias': True,
                'use_item_bias': True,
                'use_global_bias': True,
                'dropout': 0.0,
                'activation': None,
                'sparse': False
            },
            'optimizer': {
                'name': 'Adam',
                'lr': 0.01,
                'weight_decay': 0.0
            },
            'training': {
                'loss_function': 'bpr_loss',
                'n_iter': 100,
                'batch_size': 1000,
                'eval_every': 5,
                'eval_user_size': None,
                'early_stopping_patience': 10,
                'log_level': 1
            },
            'data': {
                'cache_dir': None,
                'item_feature': 'metadata',
                'neg_option': 'neg-ignore',
                'rating_threshold': 3.5,
                'train_ratio_pos': 0.8,
                'train_ratio_neg': 0.8
            },
            'mlflow': {
                'experiment_name': 'movielens_pipeline',
                'tracking_uri': 'sqlite:///mlflow.db'
            },
            'output': {
                'output_dir': './output'
            },
            'grid_search': {
                'enabled': False,
                'param_grid': {
                    'model.n_latent': [32, 64, 128],
                    'optimizer.lr': [0.001, 0.01],
                    'training.loss_function': ['bpr_loss', 'hinge_loss']
                }
            }
        }

    @staticmethod
    def save_config(config: Dict, output_path: str):
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(
                config, f, default_flow_style=False, sort_keys=False
            )
        print(f"Configuration saved to: {output_path}")
