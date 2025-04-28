"""
Recommendation system implementation using hybrid matrix factorization.

This module provides the RecSys class for training and evaluating
recommendation models based on user-item interaction data.

Author: Rimple Sandhu
Email: rimple.sandhu@outlook.com
"""

import os
import sys
import json
import logging
from typing import List, Optional, Union, Tuple, Dict, Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
import torch
from tqdm import tqdm

from .dataset import UserItemData
from .hybrid_mf import HybridMF
from .utils import scipy_csr_to_torch_csr, random_col_for_row, sample_pos_neg_pairs

# Configure logger
logger = logging.getLogger(__name__)


class RecSys:
    """Base class for recommendation systems using hybrid matrix factorization."""

    def __init__(
        self,
        data: UserItemData,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable,
        output_dir: str,
        log_level: int = logging.INFO,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        """Initialize the recommendation system.

        Args:
            data: User-item interaction data
            model: Custom model for scoring user-item pair
            optimizer: PyTorch optimizer class
            log_level: Logging level (default: logging.INFO)
            random_state: Random seed or RandomState for reproducibility
            output_dir: Directory to save metrics and plots (optional)
        """
        # Configure logger
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)

        # Add handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.info('Initiating Hybrid Recommender System..')

        # Set output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f'Output directory set to: {output_dir}')

        # Initialize random state
        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)

        # user item data
        self.data: UserItemData = data
        self.data.validate_dataset()

        # model
        self.model: torch.nn.Module = model
        self.optimizer: torch.optim.Optimizer = optimizer(
            self.model.parameters())
        self.loss_function = loss_function
        self._check_compatibility_model_data()
        self.metrics = []

        # csr matrices
        self.Rpos_train_csr = self.data.Rpos_train.tocsr()
        self.Rneg_train_csr = self.data.Rneg_train.tocsr()
        self.Rpos_test_csr = self.data.Rpos_test.tocsr()
        self.Rneg_test_csr = self.data.Rneg_test.tocsr()
        self.Fu_csr = self.data.Fu.tocsr()
        self.Fi_csr = self.data.Fi.tocsr()

        # valid train users (only those that have atleast 1 pos/neg)
        self.users_train = list(set(self.data.Rpos_train.row))
        self.users_test = list(set(self.data.Rpos_test.row))

    def __repr__(self) -> str:
        """String representation of the RecSys object."""
        return (
            f'{self.__class__.__name__}(\n'
            f'Data={self.data.__repr__()}\n'
            f'Model={self.model.__repr__()}\n'
            f'Optimizer={self.optimizer.__repr__()}\n)'
        )

    def _check_compatibility_model_data(self):
        """check if model compatible with data"""
        # Check matching dimensions
        if self.data.n_user_features != self.model.n_user_features:
            istr = (f"User feature dimension mismatch:"
                    f"data={self.data.n_user_features}, "
                    f"model={self.model.n_user_features}, ")
            self.logger.error(istr)
            raise ValueError(istr)

        if self.data.n_item_features != self.model.n_item_features:
            istr = (f"Item feature dimension mismatch:"
                    f"data={self.data.n_item_features}, "
                    f"model={self.model.n_item_features}, ")
            self.logger.error(istr)
            raise ValueError(istr)

    def get_model_predictions(
            self,
            batch_users,
            Rpos_csr,
            Rneg_csr,
            explicit_only
    ):
        """get model predictions"""
        users, pos_items, neg_items = sample_pos_neg_pairs(
            pos_csr_mat=Rpos_csr,
            neg_csr_mat=Rneg_csr,
            user_indices=batch_users,
            explicit_only=explicit_only
        )

        # Prepare feature embedding matrices
        Fu_sliced = scipy_csr_to_torch_csr(self.Fu_csr[users, :])
        Fi_sliced_pos = scipy_csr_to_torch_csr(self.Fi_csr[pos_items, :])
        Fi_sliced_neg = scipy_csr_to_torch_csr(self.Fi_csr[neg_items, :])

        # get predict scores
        r_ui = self.model(
            user_features=Fu_sliced,
            item_features=Fi_sliced_pos
        )
        r_uj = self.model(
            user_features=Fu_sliced,
            item_features=Fi_sliced_neg
        )
        return r_ui, r_uj

    def evaluate(
        self,
        max_users: int,
        use_train: bool,
        explicit_only
    ) -> Dict[str, float]:
        """Evaluate AUC score and loss for a set of users"""

        # Select appropriate datasets
        Rpos_csr = self.Rpos_train_csr if use_train else self.Rpos_test_csr
        Rneg_csr = self.Rneg_train_csr if use_train else self.Rneg_test_csr
        user_set = self.users_train if use_train else self.users_test

        # get random set of users
        self.model.eval()
        n_users = min(max_users, len(user_set))
        user_set_rnd = self.random_state.choice(
            a=user_set,
            size=n_users,
            replace=False
        ).tolist()

        # Process users in batches
        metrics = {}
        with torch.no_grad():
            r_ui, r_uj = self.get_model_predictions(
                user_set_rnd, Rpos_csr, Rneg_csr, explicit_only
            )
            n_users = len(r_ui)

            # metrics
            true_labels = np.zeros(n_users+n_users)
            true_labels[:n_users] = 1
            pred_labels = np.concatenate((
                r_ui.detach().cpu().numpy(),
                r_uj.detach().cpu().numpy()
            ))
            metrics['loss'] = self.loss_function(r_ui, r_uj).mean().item()
            metrics['auc'] = roc_auc_score(
                y_true=true_labels,
                y_score=pred_labels
            ).item()

            metrics['n_users_evaluated'] = n_users

        return metrics

    def fit(
        self,
        n_iter: int,
        batch_size: int = 1000,
        eval_every: int = None,
        eval_user_size: int = 1000,
        save_every: int = None,
        explicit_ns_for_train: bool = False,
        explicit_ns_for_test: bool = False,
        disable_progress_bar: bool = False
    ) -> None:
        """Train the recommendation model using BPR optimization."""

        # Adjust batch size if needed
        n_users = len(self.users_train)
        if batch_size >= n_users:
            self.logger.warning(
                f'Batch size({batch_size:,})>n_users({n_users:,})')
            batch_size = n_users
        n_mbatches = int(np.ceil(n_users / batch_size))
        self.logger.info(f'Batch size = {batch_size:,}')
        self.logger.info(f'# of minibatches = {n_mbatches:,}')

        # Set up evaluation and save frequency
        eval_frequency = eval_every if eval_every else max(1, n_iter // 10)
        save_frequency = save_every if save_every else eval_frequency
        self.logger.info(f'Eval frequency = {eval_frequency} epochs')
        self.logger.info(f'Save frequency = {save_frequency} epochs')

        # Set up progress tracking
        epoch0 = self.metrics[-1]['epoch'] if self.metrics else 0
        epoch_looper = tqdm(
            iterable=range(epoch0+1, epoch0+n_iter+1),
            total=n_iter,
            file=sys.stdout,
            desc='HybBPR',
            disable=disable_progress_bar
        )

        # device = next(self.model.parameters()).device  # Get model device
        for k, epoch in enumerate(epoch_looper):
            # get batched users
            shuffled_users = self.users_train.copy()
            self.random_state.shuffle(shuffled_users)
            batches = np.array_split(shuffled_users, n_mbatches)
            epoch_loss = 0.0
            n_users_trained = 0

            for batch_users in batches:
                # Get sliced feature matrices
                self.model.train()
                self.optimizer.zero_grad()
                r_ui, r_uj = self.get_model_predictions(
                    batch_users=batch_users,
                    Rpos_csr=self.Rpos_train_csr,
                    Rneg_csr=self.Rneg_train_csr,
                    explicit_only=explicit_ns_for_train
                )
                loss = self.loss_function(r_ui, r_uj)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(r_ui)
                n_users_trained += len(r_ui)

            # Calculate average loss and track metrics
            avg_loss = epoch_loss / n_users_trained
            metrics_dict = {'epoch': epoch, 'loss': avg_loss}

            # Evaluate on random subset of users if it's time
            if epoch % eval_frequency == 0:
                self.model.eval()
                with torch.no_grad():
                    test_metrics = self.evaluate(
                        use_train=False,
                        max_users=eval_user_size,
                        explicit_only=explicit_ns_for_test
                    )
                    train_metrics = self.evaluate(
                        use_train=True,
                        max_users=eval_user_size,
                        explicit_only=explicit_ns_for_train
                    )
                self.model.train()

                for k, v in test_metrics.items():
                    metrics_dict[f'test_{k}'] = v

                for k, v in train_metrics.items():
                    metrics_dict[f'train_{k}'] = v

                self.logger.info(
                    # f"Train AUC: {train_metrics.get('auc_mean', 0):.4f}, "
                    f"Eval at epoch {epoch}: "
                    f"Test AUC: {test_metrics.get('auc', 0):.4f}, "
                    f"Test Loss: {test_metrics.get('loss', 0):.4f}"
                )

            # Store metrics and update progress bar
            self.metrics.append(metrics_dict)
            if epoch % save_frequency == 0 or k == n_iter:
                self.save_metrics()
                self.save_model()

            # Update progress bar
            epoch_looper.set_postfix({'loss': f'{avg_loss:.4f}'})

    def save_metrics(self, filename: str = "metrics.json") -> None:
        """Save training metrics to a JSON file."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        self.logger.debug(f"Saved metrics to {filepath}")

    def save_model(self, filename: str = "rec_model.torch") -> str:
        """Save model state to file"""
        filepath = os.path.join(self.output_dir, filename)
        try:
            # Save just the model state dictionary instead of the entire model
            torch.save(self.model.state_dict(), filepath,
                       _use_new_zipfile_serialization=False, pickle_protocol=4)
            self.logger.debug(f"Model state dict saved to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to save model to {filepath}: {str(e)}")
            raise

    @classmethod
    def plot_metrics(
        cls,
        metrics_filepath: str,
        output_dir: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        title: str = 'Train/Test Metrics'
    ) -> Tuple[Optional[plt.Figure], Optional[Dict[str, plt.Axes]], pd.DataFrame]:
        """Plot training and evaluation metrics from a saved metrics file.

        Creates a single plot with dual y-axes:
        - Left y-axis: Loss values
        - Right y-axis: AUC values

        Args:
            metrics_filepath: Path to the JSON file containing metrics
            output_dir: Directory to save plots (defaults to same directory as
                    metrics file)
            figsize: Figure size as (width, height) in inches
            dpi: Resolution of the figure
            title: Title of the plot

        Returns:
            Tuple containing:
            - Figure object
            - Dictionary of Axes objects
            - DataFrame containing the metrics
        """
        # Set up logger
        logger = logging.getLogger(f"{__name__}.{cls.__name__}.plot_metrics")

        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(metrics_filepath)
        os.makedirs(output_dir, exist_ok=True)

        # Load metrics from file
        try:
            with open(metrics_filepath, 'r') as f:
                metrics_list = json.load(f)
        except Exception as e:
            logger.error(
                f"Failed to load metrics from {metrics_filepath}: {str(e)}")
            raise

        if not metrics_list:
            logger.warning(f"No metrics found in {metrics_filepath}")
            return None, None, pd.DataFrame()

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(metrics_list)
        if 'epoch' not in df.columns:
            logger.warning(f"No epoch data found in metrics")
            return None, None, df

        # Create figure with dual y-axes
        fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
        ax2 = ax1.twinx()

        # Set up plot styling
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('BPR Loss', color='tab:blue')
        ax2.set_ylabel('ROC AUC', color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax1.grid(True, alpha=0.3)

        # Start x-axis from 0
        min_epoch = 0
        max_epoch = df['epoch'].max()
        ax1.set_xlim(min_epoch, max_epoch * 1.)

        # plot
        plot_tuple = [
            ('train_loss', 'Train Loss', 'b-', ax1),
            ('test_loss', 'Test Loss', 'b--', ax1),
            ('train_auc', 'Train AUC', 'r-', ax2),
            ('test_auc', 'Test AUC', 'r--', ax2),
        ]
        lines, labels = [], []
        for ituple in plot_tuple:
            name, label, ptype, iax = ituple
            print(name)
            if name in df.columns:
                idf = df[df[name].notna()]
                line, = iax.plot(idf['epoch'], idf[name], ptype)
                line.set_label(label)
                lines.append(line)
                labels.append(label)

        # Add legend to the right side of the plot
        if lines:
            fig.legend(
                lines, labels,
                loc='center right',
                bbox_to_anchor=(1.15, 0.5)
            )

        plt.title(title)
        # Add extra right margin to accommodate annotations and prevent overlap
        plt.tight_layout(rect=[0, 0, 0.99, 1])

        # Save figure
        metrics_plot_path = os.path.join(output_dir, 'metrics_plot.png')
        plt.savefig(metrics_plot_path, bbox_inches='tight')
        logger.info(f"Saved metrics plot to {metrics_plot_path}")

        # Return figure, axes, and metrics dataframe
        axes = {'loss': ax1, 'auc': ax2}
        return fig, axes, df

# Example usage:
# plot_metrics("path/to/metrics.json")

    # def predict(
    #     self,
    #     user_ids: Optional[List[int]] = None,
    #     item_ids: Optional[List[int]] = None
    # ) -> torch.Tensor:
    #     """Predict scores for user-item pairs."""
    #     # Switch to evaluation mode
    #     self.model.train(False)

    #     # Process user indices
    #     if user_ids is None:
    #         Fu = csr_scipy_to_torch(self.Fu_csr)
    #     else:
    #         self.data._validate_user_indices(user_ids)
    #         Fu = csr_scipy_to_torch(self.Fu_csr[user_ids])

    #     # Process item indices
    #     if item_ids is None:
    #         Fi = csr_scipy_to_torch(self.Fi_csr)
    #     else:
    #         self.data._validate_item_indices(item_ids)
    #         Fi = csr_scipy_to_torch(self.Fi_csr[item_ids])

    #     # Get model parameters
    #     Eu = self.model.user_latent.weight
    #     Ei = self.model.item_latent.weight
    #     Bi = self.model.item_biases.weight

    #     # Matrix factorization calculation
    #     FuEu = torch.sparse.mm(Fu, Eu)  # Nu * Nl
    #     FiEi = torch.sparse.mm(Fi, Ei)  # Ni * Nl
    #     FiBi = torch.sparse.mm(Fi, Bi)  # Ni * 1

    #     # Prediction = user factors * item factors + item bias
    #     out = FuEu @ FiEi.T + FiBi.T
    #     return out.detach()  # Nu * Ni

    # def compute_roc_auc_score(
    #     self,
    #     uid: int,
    #     use_test_data: bool = False
    # ) -> Optional[float]:
    #     """Compute ROC AUC score for a single user."""
    #     # Get positive items based on whether we're evaluating on train or test
    #     if use_test_data:
    #         # For test set evaluation, positive items come from test set
    #         pos_matrix = self.data.Rpos_test.tocsr()
    #         if pos_matrix[uid].nnz == 0:
    #             return None
    #         item_ids_pos = pos_matrix[uid].indices
    #     else:
    #         # For training set evaluation, use training data
    #         item_ids_pos = self.Rpos_train_lil.rows[uid]

    #     # Negative items are the same for both train and test evaluation
    #     item_ids_neg = self.Rneg_lil.rows[uid]

    #     # Only evaluate users with sufficient interactions
    #     min_interactions = 5
    #     if len(item_ids_pos) > min_interactions and len(item_ids_neg) > min_interactions:
    #         # Combine positive and negative items for evaluation
    #         item_ids = np.concatenate((item_ids_pos, item_ids_neg))

    #         # Create binary ground truth labels (1=positive, 0=negative)
    #         ytrue_binary = np.zeros(len(item_ids), dtype=int)
    #         ytrue_binary[:len(item_ids_pos)] = 1

    #         # Get model predictions for all items
    #         ypred = self.predict(
    #             user_ids=[uid],
    #             item_ids=item_ids
    #         ).squeeze().tolist()

    #         # Convert prediction scores to binary recommendations
    #         # using top-k approach where k is the number of positive items
    #         k = len(item_ids_pos)
    #         top_k_indices = np.argsort(ypred)[::-1][:k]
    #         ypred_binary = np.zeros(len(item_ids), dtype=int)
    #         ypred_binary[top_k_indices] = 1

    #         # Compute and return AUC score
    #         return sklearn.metrics.roc_auc_score(ytrue_binary, ypred_binary)

    #     return None

    # def _get_sliced_feature_matrices(
    #     self,
    #     batch_users: np.ndarray,
    #     Rpos_csr: sp.csr_matrix,
    #     Rneg_csr: sp.csr_matrix
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """Return sliced featured embedding matrices """
    #     # Get positive and negative items for this batch
    #     pos_item_ids = random_col_for_row(Rpos_csr, batch_users)
    #     neg_item_ids = random_col_for_row(Rneg_csr, batch_users)

    #     # Prepare feature embedding matrices
    #     Fu_sliced = scipy_csr_to_torch_csr(self.Fu_csr[batch_users, :])
    #     Fi_sliced_pos = scipy_csr_to_torch_csr(self.Fi_csr[pos_item_ids, :])
    #     Fi_sliced_neg = scipy_csr_to_torch_csr(self.Fi_csr[neg_item_ids, :])

    #     return Fu_sliced, Fi_sliced_pos, Fi_sliced_neg

    # def _log(self, message: str) -> None:
    #     """Log a message if verbose mode is enabled."""
    #     if self.verbose:
    #         print(f'{self.__class__.__name__}: {message}', flush=True)

    # def _evaluate_model(self, users: np.ndarray, epoch: int, sample_size: int = 100) -> None:
    #     """Evaluate the model on both training and test datasets."""
    #     # Sample users for evaluation (or use all if fewer than sample_size)
    #     if len(users) > sample_size:
    #         eval_users = np.random.choice(users, sample_size, replace=False)
    #     else:
    #         eval_users = users

    #     # Initialize metrics
    #     train_aucs = []
    #     test_aucs = []

    #     # Calculate metrics efficiently using vectorized operations where possible
    #     for uid in eval_users:
    #         # Calculate training AUC
    #         train_auc = self.compute_roc_auc_score(uid, use_test_data=False)
    #         if train_auc is not None:
    #             train_aucs.append(train_auc)

    #         # Calculate test AUC
    #         test_auc = self.compute_roc_auc_score(uid, use_test_data=True)
    #         if test_auc is not None:
    #             test_aucs.append(test_auc)

    #     # Store metrics
    #     if train_aucs:
    #         self.metrics['train_auc']['mean'].append(np.mean(train_aucs))
    #         self.metrics['train_auc']['std'].append(np.std(train_aucs))
    #         self.metrics['train_auc']['count'].append(len(train_aucs))

    #     if test_aucs:
    #         self.metrics['test_auc']['mean'].append(np.mean(test_aucs))
    #         self.metrics['test_auc']['std'].append(np.std(test_aucs))
    #         self.metrics['test_auc']['count'].append(len(test_aucs))

    #     if self.verbose:
    #         train_msg = f"{len(train_aucs)} users, mean: {np.mean(train_aucs):.4f}" if train_aucs else "No valid users"
    #         test_msg = f"{len(test_aucs)} users, mean: {np.mean(test_aucs):.4f}" if test_aucs else "No valid users"
    #         self._log(
    #             f"Epoch {epoch}: Train AUC ({train_msg}), Test AUC ({test_msg})")

    # def plot_metrics(self, figsize=(12, 8)) -> object:
    #     """Plot training metrics."""
    #     try:
    #         import matplotlib.pyplot as plt
    #         import pandas as pd

    #         df = self.get_metrics_dataframe()

    #         fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    #         # Plot loss
    #         axes[0].plot(df['epoch'], df['loss'], 'b-', label='Training Loss')
    #         axes[0].set_ylabel('Loss')
    #         axes[0].set_title('Training Loss')
    #         axes[0].grid(True)

    #         # Plot AUC metrics if available
    #         if 'train_auc_mean' in df.columns or 'test_auc_mean' in df.columns:
    #             if 'train_auc_mean' in df.columns:
    #                 axes[1].plot(df['epoch'], df['train_auc_mean'], 'g-',
    #                              label='Train AUC')
    #             if 'test_auc_mean' in df.columns:
    #                 axes[1].plot(df['epoch'], df['test_auc_mean'], 'r-',
    #                              label='Test AUC')

    #             axes[1].set_xlabel('Epoch')
    #             axes[1].set_ylabel('AUC Score')
    #             axes[1].set_title('AUC Evaluation')
    #             axes[1].grid(True)
    #             axes[1].legend()

    #         plt.tight_layout()
    #         return fig

    #     except ImportError:
    #         self._log(
    #             "Matplotlib and/or pandas not available. Cannot plot metrics.")
    #         return None


# """
# Base class for defining User-Item interaction data

# Author: Rimple Sandhu
# Email: rimple.sandhu@outlook.com
# """

# import sys
# import numpy as np
# from numpy import ndarray
# from scipy.sparse import find, sparray
# import torch
# from .dataset import UserItemData
# from .hybrid_mf import HybridMF
# from .utils import csr_scipy_to_torch
# from tqdm import tqdm
# import sklearn
# # pylint: disable=invalid-name

    # def get_pos_neg_pairs(self, user_ids: List[int]) -> Tuple[List[int], List[int]]:
    #     """Generate positive and negative item pairs for each user."""
    #     # Get positive and negative interaction rows for each user
    #     pos_interactions = self.Rpos_train_lil[user_ids]
    #     neg_interactions = self.Rneg_lil[user_ids]
    #     pos_item_ids = []
    #     neg_item_ids = []

    #     # For each user, select one positive and one negative item
    #     for plist, nlist in zip(pos_interactions.rows, neg_interactions.rows):
    #         # Select random positive item
    #         pos_item_ids.append(self.rstate.choice(plist))

    #         # Select random negative item (or generate one if needed)
    #         if nlist:
    #             neg_item_ids.append(self.rstate.choice(nlist))
    #         else:
    #             # If no explicit negative items, sample a random item
    #             # that is not in the positive list
    #             item_id_rnd = self.rstate.randint(self.data.n_items)
    #             while item_id_rnd in plist:
    #                 item_id_rnd = self.rstate.randint(self.data.n_items)
    #             neg_item_ids.append(item_id_rnd)

    #     return pos_item_ids, neg_item_ids


# class RecSys:
#     """
#     Base class for setting up data for recomednation system
#     """

#     def __init__(
#         self,
#         ui_data: UserItemData,
#         n_latent: int,
#         optimizer: torch.optim.Optimizer,
#         verbose: bool = False
#     ):
#         """Initiate"""
#         self.verbose = verbose
#         self.printit('Initiating Hybrid Recomender System..')
#         self.rstate = np.random.RandomState()
#         self.data = ui_data
#         self.model = HybridMF(
#             n_item_features=self.data.n_item_features,
#             n_user_features=self.data.n_user_features,
#             n_latent=n_latent
#         )
#         self.optimizer = optimizer(self.model.parameters())
#         self.loss_tracker = []
#         self.auc_tracker = []

#         # extract pos/neg interactions
#         self.Rpos_train_coo: sparray = self.data.Rpos_train_coo
#         self.Rpos_test_coo: sparray = self.data.Rpos_test_coo
#         self.Rneg_coo: sparray = self.data.Rneg_coo
#         self.Rpos_train_lil: sparray = self.data.Rpos_train_coo.tolil()
#         self.Rpos_test_csr: sparray = self.Rpos_test_coo.tocsr()
#         self.Rneg_lil: sparray = self.Rneg_coo.tolil()
#         self.Fu_csr: sparray = self.data.Fu_coo.tocsr()
#         self.Fi_csr: sparray = self.data.Fi_coo.tocsr()

#         # # print some info
#         # ndata = self.Rpos_train_coo.nnz
#         # self.printit(f'Got {self.Rpos_coo.nnz:,}/{ndata:,} +ve interactions')
#         # self.printit(f'Got {self.Rneg_coo.nnz:,}/{ndata:,} -ve interactions')

#         # # find negative ineractions
#         # if len(pos_inds) == len(wgts):
#         #     self.user_ids_neg = []
#         #     self.item_ids_neg = []
#         #     self.wgts_neg = []
#         # else:
#         #     neg_inds = [ix for ix in range(len(wgts)) if ix not in pos_inds]
#         #     self.user_ids_neg = rows[neg_inds]
#         #     self.item_ids_neg = cols[neg_inds]
#         #     self.wgts_neg = wgts[neg_inds]

#     def fit(
#             self,
#             n_iter: int,
#             rng_seed: int = 1234,
#             batch_size: int = 1000,
#             explicit_neg_sampling: bool = True,
#             disable_progress_bar: bool = False
#     ):
#         """Fitting model"""

#         # random number generator
#         self.printit(f'Setting rng seed to {rng_seed}')
#         self.rstate = np.random.RandomState(rng_seed)

#         # batch size for minibatching
#         valid_users = np.unique(self.Rpos_train_coo.row)
#         valid_users_test = np.unique(self.Rpos_test_coo.row)
#         n_users = len(valid_users)
#         self.printit(f'{n_users:,}/{self.data.n_users:,} have +ve int data')
#         if batch_size >= n_users:
#             self.printit(
#                 f'Batch size({batch_size:,})'
#                 f'>n_users({n_users:,}). '
#             )
#             batch_size = n_users
#         n_mbatches = int(n_users/batch_size)
#         self.printit(f'Batch size = {batch_size:,}')
#         self.printit(f'# of minibatches = {n_mbatches:,}')

#         # get users with pos interactions
#         pos_ids_randomized = list(range(n_users))
#         epoch_looper = tqdm(
#             iterable=range(n_iter),
#             total=n_iter,
#             # position=0,
#             # leave=True,
#             file=sys.stdout,
#             desc='HybBPR',
#             disable=disable_progress_bar
#         )

#         for epoch in epoch_looper:

#             # randomlized positive indices at every epoch
#             self.rstate.shuffle(pos_ids_randomized)
#             eloss = 0.
#             test_loss = 0.
#             for ids in np.array_split(pos_ids_randomized, n_mbatches):

#                 # user, pos item, neg item pairs
#                 # pos_user_ids = self.Rpos_coo.row[ids]
#                 self.model.train(True)
#                 self.optimizer.zero_grad()

#                 user_ids = valid_users[ids]
#                 pos_item_ids, neg_item_ids = self.get_pos_neg_pairs(user_ids)
#                 # pos_item_ids = self.Rpos_coo.col[ids]
#                 # neg_item_ids = self.get_neg_samples(user_ids=pos_user_ids)
#                 # pos_wgts = self.Rpos_coo.data[ids]

#                 # Feature matrices
#                 Fu_sliced = csr_scipy_to_torch(self.Fu_csr[user_ids])
#                 Fi_sliced_pos = csr_scipy_to_torch(self.Fi_csr[pos_item_ids])
#                 Fi_sliced_neg = csr_scipy_to_torch(self.Fi_csr[neg_item_ids])

#                 # predictions from the model
#                 r_uij = self.model.forward(
#                     Fuser=Fu_sliced,
#                     Fitem_pos=Fi_sliced_pos,
#                     Fitem_neg=Fi_sliced_neg
#                 )

#                 loss = (1.0 - torch.sigmoid(r_uij)).mean()
#                 eloss += loss.item()
#                 loss.backward()
#                 self.optimizer.step()
#                 epoch_looper.refresh()
#                 # self.printit(f'Epoch-{epoch},Ndata={len(ids)},Eloss={eloss}')

#             # auc score
#             list_of_auc = []
#             for uid in np.random.choice(valid_users, 100):
#                 auc_score = self.compute_roc_auc_score(uid)
#                 if auc_score is not None:
#                     list_of_auc.append(auc_score)
#             self.auc_tracker.append((
#                 len(list_of_auc),
#                 np.mean(list_of_auc),
#                 np.std(list_of_auc)
#             ))

#             eloss /= n_mbatches
#             # self.printit(
#             # f"Epoch [{epoch+1}/{n_iter}] Loss: {np.round(eloss, 4)}")
#             self.loss_tracker.append(eloss)

#     def get_pos_neg_pairs(self, user_ids):
#         """negative sampling"""
#         Rneg_sliced_lil = self.Rneg_lil[user_ids]
#         Rpos_sliced_lil = self.Rpos_train_lil[user_ids]
#         pos_item_ids = []
#         neg_item_ids = []
#         for plist, nlist in zip(Rpos_sliced_lil.rows, Rneg_sliced_lil.rows):
#             pos_item_ids.append(self.rstate.choice(plist))
#             if nlist:
#                 neg_item_ids.append(self.rstate.choice(nlist))
#             else:
#                 item_id_rnd = self.rstate.randint(self.data.n_items)
#                 while item_id_rnd in plist:
#                     item_id_rnd = self.rstate.randint(self.data.n_items)
#                 neg_item_ids.append(item_id_rnd)
#         return pos_item_ids, neg_item_ids

#     def predict(
#         self,
#         user_ids: list[int] | None = None,
#         item_ids: list[int] | None = None
#     ):
#         """Predict score"""
#         # user indices
#         self.model.train(False)
#         if user_ids is None:
#             Fu = csr_scipy_to_torch(self.Fu_csr)
#         else:
#             self.data._validate_user_indices(user_ids)
#             Fu = csr_scipy_to_torch(self.Fu_csr[user_ids])

#         # item indices
#         if item_ids is None:
#             Fi = csr_scipy_to_torch(self.Fi_csr)
#         else:
#             self.data._validate_item_indices(item_ids)
#             # print(len(item_ids))
#             Fi = csr_scipy_to_torch(self.Fi_csr[item_ids])

#         # matrix factorization
#         Eu = self.model.user_latent.weight
#         Ei = self.model.item_latent.weight
#         Bi = self.model.item_biases.weight

#         FuEu = torch.sparse.mm(Fu, Eu)  # Nu * Nl
#         FiEi = torch.sparse.mm(Fi, Ei)  # Ni * Nl
#         FiBi = torch.sparse.mm(Fi, Bi)  # Ni * 1
#         out = FuEu @ FiEi.T + FiBi.T
#         return out.detach()  # Nu * Ni

#     def compute_roc_auc_score(self, uid: int):
#         """ROC AUC score"""
#         item_ids_pos = self.Rpos_train_lil.rows[uid]
#         item_ids_neg = self.Rneg_lil.rows[uid]
#         if (len(item_ids_pos) > 5) & (len(item_ids_neg) > 5):
#             item_ids = np.concatenate((item_ids_pos, item_ids_neg))
#             ytrue_binary = np.zeros(len(item_ids), dtype=int)
#             ytrue_binary[:len(item_ids_pos)] = 1
#             ypred = self.predict(
#                 user_ids=[uid],
#                 item_ids=item_ids
#             ).squeeze().tolist()
#             inds_sorted = np.argsort(ypred)[::-1][:len(item_ids_pos)]
#             ypred_binary = np.zeros(len(item_ids), dtype=int)
#             ypred_binary[inds_sorted] = 1
#             # print('Ytrue: ', ytrue_binary)
#             # print('Ypred: ', ypred_binary)
#             score = sklearn.metrics.roc_auc_score(
#                 ytrue_binary,
#                 ypred_binary
#             )
#             # print('AUC Score: ', score)
#             return score
#         return None

#     def value_err(self, istr:
#                   str) -> None:
#         """Raise value Error"""
#         raise ValueError(f'{self.__class__.__name__}: {istr}')

#     def printit(self, istr: str) -> None:
#         """Raise value Error"""
#         if self.verbose:
#             print(f'{self.__class__.__name__}: {istr}', flush=True)

#     def __repr__(self):
#         return (f'{self.__class__.__name__}(\n'
#                 f'data={self.data.__repr__()}\n'
#                 f'model={self.model.__repr__()}\n'
#                 f'optimizer={self.optimizer.__repr__()}\n)'
#                 )

#     # def _get_padded_feature_data(
#     #         self,
#     #         ids: list[int] | ndarray,
#     #         is_user: bool
#     # ):
#     #     """Get padded tensor containing feature ids for selected users"""

#     #     # Get sliced feature matrix
#     #     Fmat = self.data.Fu_coo if is_user else self.data.Fi_coo
#     #     Fmat_sliced = Fmat.tolil()[ids]

#     #     # features indices as padded tensor
#     #     feat_ids_ts = [torch.IntTensor(ix) for ix in Fmat_sliced.rows]
#     #     feat_ids_ts = torch.nn.utils.rnn.pad_sequence(feat_ids_ts).T

#     #     # feature wgts as padded tensor
#     #     feat_wgts_ts = [torch.Tensor(ix) for ix in Fmat_sliced.data]
#     #     feat_wgts_ts = torch.nn.utils.rnn.pad_sequence(feat_wgts_ts).T

#     #     return feat_ids_ts, feat_wgts_ts

#         # neg_indices = None
#         # if explicit_neg_sampling:
#         #     if self.user_ids_neg:
#         #         neg_indices = list(range(len(self.user_ids_neg)))

#         # # randomlized positive indices at every epoch
#         # if neg_indices:
#         #     rstate.shuffle(neg_indices)
#         #     uid_neg_randomized = self.user_ids_neg[neg_indices]
#         #     iid_neg_randomized = self.item_ids_neg[neg_indices]
#         #    user_feat_ids, user_feat_wgts = self._get_padded_feature_data(
#         #        ids=pos_user_ids,
#         #        is_user=True
#         #    )
#         #    item_feat_ids_pos, item_feat_wgts_pos = self._get_padded_feature_data(
#         #        ids=pos_item_ids,
#         #        is_user=False
#         #    )
#         #    item_feat_ids_neg, item_feat_wgts_neg = self._get_padded_feature_data(
#         #        ids=neg_item_ids,
#         #        is_user=False
#         #    )

#         #    # Pass it rhough the model
#         #    pos_preds = self.model.forward(
#         #        user_feature_indices=user_feat_ids,
#         #        item_feature_indices=item_feat_ids_pos,
#         #        user_feature_weights=user_feat_wgts,
#         #        item_feature_weights=item_feat_wgts_pos
#         #    )

#         #    neg_preds = self.model.forward(
#         #        user_feature_indices=user_feat_ids,
#         #        item_feature_indices=item_feat_ids_neg,
#         #        user_feature_weights=user_feat_wgts,
#         #        item_feature_weights=item_feat_wgts_neg
#         #    )


#  batch_aucs = []
#                 batch_loss = 0.0
#                 valid_batch_users = []
#                 # here need to use the new function in utils
#                 for user_idx in batch_users:
#                     auc = self.compute_user_auc(
#                         user_idx, Rpos_csr, Rneg_csr
#                     )
#                     if auc is not None:
#                         batch_aucs.append(auc)
#                         valid_batch_users.append(user_idx)

#                 # Calculate BPR loss and ROC AUC for valid users in batch
#                 if valid_batch_users:
#                     loss = self.compute_loss(batch_users, Rpos_csr, Rneg_csr)
#                     batch_loss = loss.mean().item()*len(valid_batch_users)
#                     total_loss += batch_loss
#                     auc_scores.extend(batch_aucs)
#                     n_users_evaluated += len(valid_batch_users)

    # def compute_user_auc(
    #     self,
    #     user_idx: int,
    #     Rpos_csr: sp.csr_matrix,
    #     Rneg_csr: sp.csr_matrix
    # ):
    #     """Compute ROC AUC score for a single user"""
    #     # Get positive item indices for this user
    #     start_pos, end_pos = Rpos_csr.indptr[user_idx:user_idx+2]
    #     user_pos_items = Rpos_csr.indices[start_pos:end_pos]

    #     # Check if we have enough positive interactions
    #     if len(user_pos_items) < 2:
    #         return None

    #     # get negative item indices for this user
    #     if Rneg_csr.nnz > 0:
    #         # Get negative item indices
    #         start_neg, end_neg = Rneg_csr.indptr[user_idx:user_idx+2]
    #         user_neg_items = Rneg_csr.indices[start_neg:end_neg]
    #         if len(user_neg_items) == 0:
    #             return None

    #     else:
    #         user_neg_items = list(set(self.items_all) - set(user_pos_items))

    #     # # enforce limits on number of negative interactions
    #     # if self.auc_params['neg_count_equal_or_lessthan_pos_count']:
    #     #     user_neg_items = self.random_state.choice(
    #     #         a=user_neg_items,
    #     #         size=min(len(user_pos_items), len(user_neg_items)),
    #     #         replace=False
    #     #     ).tolist()

    #     # Create ground truth labels (1 for positive, 0 for negative)
    #     true_labels = np.zeros(len(user_pos_items)+len(user_neg_items))
    #     true_labels[:len(user_pos_items)] = 1

    #     # Concatenate item IDs and create corresponding user IDs
    #     all_items = np.concatenate([user_pos_items, user_neg_items])
    #     all_users = np.full_like(all_items, user_idx)

    #     # Get predictions
    #     Fu_sliced = scipy_csr_to_torch_csr(self.Fu_csr[all_users, :])
    #     Fi_sliced = scipy_csr_to_torch_csr(self.Fi_csr[all_items, :])
    #     all_scores = self.model(
    #         user_features=Fu_sliced,
    #         item_features=Fi_sliced
    #     )
    #     user_auc = roc_auc_score(
    #         y_true=true_labels,
    #         y_score=all_scores.detach().cpu().numpy()
    #     )
    #     return user_auc

    # def compute_loss(self, users: List[int], pos_items: List[int], neg_items: List[int]):
    #     """computes loss"""
    #     # Prepare feature embedding matrices
    #     Fu_sliced = scipy_csr_to_torch_csr(self.Fu_csr[users, :])
    #     Fi_sliced_pos = scipy_csr_to_torch_csr(self.Fi_csr[pos_items, :])
    #     Fi_sliced_neg = scipy_csr_to_torch_csr(self.Fi_csr[neg_items, :])

    #     # get predict scores
    #     r_ui = self.model(
    #         user_features=Fu_sliced,
    #         item_features=Fi_sliced_pos
    #     )
    #     r_uj = self.model(
    #         user_features=Fu_sliced,
    #         item_features=Fi_sliced_neg
    #     )
    #     return self.loss_function(r_ui, r_uj)
