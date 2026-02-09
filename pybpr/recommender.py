"""Recommendation system with hybrid matrix factorization."""

import sys
from typing import Dict, List, Optional, Union

import mlflow.pytorch
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .losses import PairwiseLossFn
from .utils import (
    compute_auc_scores,
    get_user_interactions,
    sample_pos_neg_pairs,
    split_sparse_coo_matrix,
)


class RecommendationSystem:
    """Recommendation system using hybrid matrix factorization."""

    def __init__(
        self,
        Rpos: sp.spmatrix,
        Rneg: sp.spmatrix,
        Fu: sp.spmatrix,
        Fi: sp.spmatrix,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: PairwiseLossFn,
        device: Union[torch.device, str],
        mlflow_run: mlflow.ActiveRun,
        train_ratio: float = 0.8,
        random_state: Optional[int] = None,
    ):
        """Initialize recommendation system."""
        print('Initiating Hybrid Recommender System..')

        # MLflow tracking
        self.mlflow_run = mlflow_run
        print(
            f'MLflow run ID: '
            f'{self.mlflow_run.info.run_id}'
        )

        # Device configuration
        self.device = (
            torch.device(device)
            if isinstance(device, str) else device
        )
        print(f'Using device: {self.device}')

        # Store feature dimensions for model checks
        self.n_user_features = Fu.shape[1]
        self.n_item_features = Fi.shape[1]

        # Model setup
        self.model = model.to(self.device)
        self.optimizer: torch.optim.Optimizer = optimizer(
            self.model.parameters()
        )
        self.loss_function = loss_function
        self._check_compatibility_model_data()

        # Split positive interactions into train/test
        Rpos_train, Rpos_test = split_sparse_coo_matrix(
            Rpos, train_ratio, random_state
        )

        # Split negative interactions into train/test
        if Rneg.nnz > 0:
            Rneg_train, Rneg_test = split_sparse_coo_matrix(
                Rneg, train_ratio, random_state
            )
        else:
            shape = Rpos.shape
            Rneg_train = sp.coo_matrix(
                shape, dtype=Rpos.dtype
            )
            Rneg_test = sp.coo_matrix(
                shape, dtype=Rpos.dtype
            )

        # CSR matrices for efficient sparse operations
        self.Rpos_train_csr = Rpos_train.tocsr()
        self.Rneg_train_csr = Rneg_train.tocsr()
        self.Rpos_test_csr = Rpos_test.tocsr()
        self.Rneg_test_csr = Rneg_test.tocsr()
        self.Fu_csr = Fu.tocsr()
        self.Fi_csr = Fi.tocsr()

        # Full positive matrix for neg sampling exclusion
        self.Rpos_all_csr = Rpos.tocsr()

        # Valid users (present in both train and test)
        users_train = np.unique(Rpos_train.row)
        users_test = np.unique(Rpos_test.row)
        self.users = np.intersect1d(
            users_train, users_test
        )
        print(f'Got {len(self.users)} users for train/test')

    def __repr__(self) -> str:
        """String representation of the object."""
        return (
            f'{self.__class__.__name__}(\n'
            f'Model={self.model.__repr__()}\n'
            f'Optimizer={self.optimizer.__repr__()}\n)'
        )

    def _check_compatibility_model_data(self):
        """Check if model compatible with data."""
        # Check user feature dimensions
        if (self.n_user_features
                != self.model.n_user_features):
            err_msg = (
                f"User feature dimension mismatch: "
                f"data={self.n_user_features}, "
                f"model={self.model.n_user_features}"
            )
            print(f"ERROR: {err_msg}")
            raise ValueError(err_msg)

        # Check item feature dimensions
        if (self.n_item_features
                != self.model.n_item_features):
            err_msg = (
                f"Item feature dimension mismatch: "
                f"data={self.n_item_features}, "
                f"model={self.model.n_item_features}"
            )
            print(f"ERROR: {err_msg}")
            raise ValueError(err_msg)

    def _get_pos_neg_scores(
        self,
        users: List[int],
        pos_csr: csr_matrix,
        neg_csr: csr_matrix,
        exclude_pos_csr: csr_matrix = None,
    ):
        """Get prediction scores for pos and neg pairs."""
        # Sample positive/negative item pairs for users
        users, pos_items, neg_items = sample_pos_neg_pairs(
            pos_csr_mat=pos_csr,
            neg_csr_mat=neg_csr,
            user_indices=users,
            exclude_pos_csr=exclude_pos_csr,
        )

        # Slice feature matrices
        Fu_sliced = self.Fu_csr[users, :]
        Fi_sliced_pos = self.Fi_csr[pos_items, :]
        Fi_sliced_neg = self.Fi_csr[neg_items, :]

        # Predict scores
        r_ui = self.model(Fu_sliced, Fi_sliced_pos)
        r_uj = self.model(Fu_sliced, Fi_sliced_neg)
        return r_ui, r_uj

    def _train(self, batch_users):
        """Train on a single batch."""
        self.model.train()

        # Get positive/negative scores from training data
        r_ui, r_uj = self._get_pos_neg_scores(
            batch_users,
            self.Rpos_train_csr,
            self.Rneg_train_csr,
        )

        # Backpropagation
        self.optimizer.zero_grad()
        loss = self.loss_function(r_ui, r_uj)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_metrics(
        self,
        users: List[int],
        use_train: bool,
    ) -> Dict[str, float]:
        """Evaluate AUC score and loss for users."""
        # Select train or test datasets
        Rpos_csr = (
            self.Rpos_train_csr if use_train
            else self.Rpos_test_csr
        )
        Rneg_csr = (
            self.Rneg_train_csr if use_train
            else self.Rneg_test_csr
        )

        metrics = {}
        with torch.no_grad():
            # Compute AUC (exclude all positives from negs)
            user_interactions = get_user_interactions(
                users=users,
                pos_csr_mat=Rpos_csr,
                neg_csr_mat=Rneg_csr,
                neg_ratio=1.,
                exclude_pos_csr=self.Rpos_all_csr,
            )
            auc_scores = compute_auc_scores(
                user_interactions=user_interactions,
                predict_fn=self.predict
            )
            metrics['auc'] = (
                np.nanmean(auc_scores).item()
            )
            metrics['auc_std'] = (
                np.nanstd(auc_scores).item()
            )

            # Compute loss (exclude all positives)
            r_ui, r_uj = self._get_pos_neg_scores(
                users, Rpos_csr, Rneg_csr,
                exclude_pos_csr=self.Rpos_all_csr,
            )
            loss = self.loss_function(r_ui, r_uj)
            metrics['loss'] = loss.item()

        return metrics

    def predict(
        self, users: List[int], items: List[int]
    ):
        """Predict score for selected items and users."""
        return self.model.predict(
            user_features=self.Fu_csr[users, :],
            item_features=self.Fi_csr[items, :]
        ).detach().numpy()[0, :]

    def evaluate(self, max_users: int = None):
        """Evaluate model on random subset of users."""
        self.model.eval()
        with torch.no_grad():
            # Sample users for evaluation
            if max_users is None:
                max_users = len(self.users)
            users = np.random.choice(
                self.users,
                size=min(max_users, len(self.users)),
                replace=False
            ).tolist()

            # Test metrics
            metrics = self.compute_metrics(
                users, use_train=False
            )
            metrics_dict = {
                f'test_{k}': v
                for k, v in metrics.items()
            }

            # Train metrics
            metrics = self.compute_metrics(
                users, use_train=True
            )
            metrics_dict |= {
                f'train_{k}': v
                for k, v in metrics.items()
            }

        self.model.train()
        return metrics_dict

    def fit(
        self,
        n_iter: int,
        batch_size: int = 1000,
        eval_every: int = 5,
        eval_user_size: int = 1000,
        early_stopping_patience: int = 10
    ) -> None:
        """Train the model using BPR optimization."""
        # Detect if running in subprocess (grid search)
        import multiprocessing as mp
        is_subprocess = (
            mp.current_process().name != 'MainProcess'
        )
        num_workers = 0 if is_subprocess else 2

        # Prepare dataloader
        dataloader = DataLoader(
            TensorDataset(torch.LongTensor(self.users)),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        print(f'# of minibatches = {len(dataloader):,}')
        print(f'Eval frequency = {eval_every} epochs')

        # Log hyperparameters to MLflow
        mlflow.log_params({
            'n_iter': n_iter,
            'batch_size': batch_size,
            'eval_every': eval_every,
            'eval_user_size': eval_user_size,
            'early_stopping_patience': (
                early_stopping_patience
            ),
        })

        # Early stopping setup
        best_test_auc = 0.0
        best_epoch = 0
        patience_counter = 0

        # Progress tracking setup
        epoch_looper = tqdm(
            iterable=range(1, n_iter+1),
            total=n_iter,
            file=sys.stdout,
            desc='HybBPR',
            ncols=70,
            unit='ep'
        )

        # Training loop
        for epoch in epoch_looper:
            epoch_loss = 0.0
            n_batches = 0

            # Batch training
            for batch in dataloader:
                batch_users = batch[0].numpy()
                loss = self._train(batch_users)
                epoch_loss += loss
                n_batches += 1

            # Track average loss across batches
            avg_loss = epoch_loss / n_batches
            epoch_looper.set_postfix(
                {'loss': f'{avg_loss:.4f}'}
            )

            # Periodic evaluation
            if epoch % eval_every == 0:
                eval_metrics = self.evaluate(
                    max_users=eval_user_size
                )
                current_auc = eval_metrics.get(
                    'test_auc', 0
                )

                # Log metrics to MLflow
                mlflow.log_metrics(
                    eval_metrics, step=epoch
                )

                # Track best model
                if current_auc > best_test_auc:
                    best_test_auc = current_auc
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping check
                if (patience_counter
                        >= early_stopping_patience):
                    print(
                        f"Early stopping at "
                        f"epoch {epoch}"
                    )
                    break

                # Update progress bar with eval metrics
                epoch_looper.write(
                    f"E{epoch}: "
                    f"AUC {eval_metrics.get('train_auc',0):.3f}/"
                    f"{eval_metrics.get('test_auc',0):.3f} | "
                    f"Loss {eval_metrics.get('train_loss',0):.3f}/"
                    f"{eval_metrics.get('test_loss',0):.3f}"
                )

        # Save best model after training completes
        if best_epoch > 0:
            self.save_model(
                name=f"best_model_epoch_{best_epoch}",
                tqdm_obj=None
            )

    def save_model(
        self,
        name: str = "model",
        tqdm_obj=None
    ) -> None:
        """Save model to MLflow."""
        try:
            mlflow.pytorch.log_model(
                pytorch_model=self.model, name=name
            )
            msg = f"Logged model to MLflow: {name}"
            if tqdm_obj:
                tqdm_obj.write(msg)
            else:
                print(msg)
        except Exception as e:
            err_msg = (
                f"Failed to save model: {str(e)}"
            )
            print(f"ERROR: {err_msg}")
            raise
