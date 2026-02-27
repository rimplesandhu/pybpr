"""Streamlined recommendation system with hybrid MF."""

import sys
from typing import Callable, Dict, List, Optional, Tuple, Union

import mlflow.pytorch
import numpy as np
import torch
import torch.optim
from numpy.random import RandomState
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .interaction_data import UserItemData
from .losses import PairwiseLossFn
from .matrix_factorization import HybridMF
from .utils import get_nonempty_rows_csr


# Utility functions (moved from utils.py for self-contained module)
def sample_pos_neg_pairs(
    pos_csr_mat: csr_matrix,
    neg_csr_mat: csr_matrix,
    user_indices: List[int],
    exclude_pos_csr: Optional[csr_matrix] = None,
    random_state: Optional[Union[int, RandomState]] = None
) -> Tuple[List[int], List[int], List[int]]:
    """Sample positive-negative item pairs for users."""
    # Initialize random state
    rng = (
        RandomState(random_state) if not isinstance(
            random_state, RandomState
        ) else random_state
    )

    # Convert to LIL for efficient row access
    pos_lil_mat = pos_csr_mat.tolil()
    neg_lil_mat = neg_csr_mat.tolil()
    exclude_lil = (
        exclude_pos_csr.tolil() if exclude_pos_csr is not None
        else pos_lil_mat
    )

    # Sample pairs
    valid_user_indices = []
    pos_item_indices = []
    neg_item_indices = []
    num_items = pos_csr_mat.shape[1]

    for user_idx in user_indices:
        pos_items = pos_lil_mat.rows[user_idx]
        if len(pos_items) == 0:
            continue

        # Sample positive item
        pos_idx = pos_items[rng.randint(0, len(pos_items))]

        # Sample negative item
        neg_items = neg_lil_mat.rows[user_idx]
        if len(neg_items) > 0:
            neg_idx = neg_items[rng.randint(0, len(neg_items))]
        else:
            # Sample excluding positives
            exclude_items = exclude_lil.rows[user_idx]
            neg_idx = rng.choice(num_items)
            while neg_idx in exclude_items:
                neg_idx = rng.choice(num_items)

        valid_user_indices.append(user_idx)
        pos_item_indices.append(pos_idx)
        neg_item_indices.append(neg_idx)

    return valid_user_indices, pos_item_indices, neg_item_indices


def get_user_interactions(
    users: List[int],
    pos_csr_mat: csr_matrix,
    neg_csr_mat: csr_matrix,
    neg_ratio: float = 1.0,
    exclude_pos_csr: Optional[csr_matrix] = None,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """Collect positive and negative interactions per user."""
    n_items = pos_csr_mat.shape[1]

    # Pre-compute indices for efficient CSR access
    pos_indptr = pos_csr_mat.indptr
    pos_indices = pos_csr_mat.indices
    neg_indptr = neg_csr_mat.indptr
    neg_indices = neg_csr_mat.indices

    # Handle exclusion matrix
    if exclude_pos_csr is not None:
        excl_indptr = exclude_pos_csr.indptr
        excl_indices = exclude_pos_csr.indices
    else:
        excl_indptr = pos_indptr
        excl_indices = pos_indices

    user_interactions = []

    for user_idx in users:
        # Get positive items
        pos_start = pos_indptr[user_idx]
        pos_end = pos_indptr[user_idx + 1]
        pos_items = pos_indices[pos_start:pos_end]

        if len(pos_items) == 0:
            continue

        # Get negative items
        neg_start = neg_indptr[user_idx]
        neg_end = neg_indptr[user_idx + 1]
        neg_items = neg_indices[neg_start:neg_end]

        # Sample negatives if not explicitly provided
        if len(neg_items) == 0:
            excl_start = excl_indptr[user_idx]
            excl_end = excl_indptr[user_idx + 1]
            excl_items = set(excl_indices[excl_start:excl_end])

            n_neg = int(neg_ratio * len(pos_items))
            n_neg = min(n_neg, n_items - len(excl_items))
            if n_neg <= 0:
                continue

            # Sample negatives excluding positives
            neg_items = np.empty(n_neg, dtype=np.int32)
            sampled = 0
            while sampled < n_neg:
                batch_size = min(n_neg - sampled, n_neg)
                candidates = np.random.randint(
                    0, n_items, size=batch_size
                )
                for item in candidates:
                    if item not in excl_items:
                        neg_items[sampled] = item
                        sampled += 1
                        if sampled >= n_neg:
                            break

        user_interactions.append(
            (user_idx, pos_items, neg_items)
        )

    return user_interactions


def compute_auc_scores(
    user_interactions: List[Tuple[int, np.ndarray, np.ndarray]],
    predict_fn: Callable
) -> List[float]:
    """Calculate ROC AUC scores using user interactions."""
    all_scores = []

    for user_idx, pos_items, neg_items in user_interactions:
        # Prepare items for batch prediction
        all_items = np.concatenate([pos_items, neg_items])

        # Predict scores
        all_predictions = predict_fn(user_idx, all_items)

        # Create binary labels
        y_true = np.zeros(len(all_items), dtype=np.int8)
        y_true[:len(pos_items)] = 1

        # Compute AUC
        user_auc = roc_auc_score(y_true, all_predictions)
        all_scores.append(user_auc)

    return all_scores


class RecommendationSystem:
    """Streamlined recommendation system using hybrid MF."""

    def __init__(
        self,
        uidata: UserItemData,
        model: HybridMF,
        optimizer: Callable[..., torch.optim.Optimizer],
        loss: PairwiseLossFn,
        device: Union[torch.device, str],
        mlflow_run: mlflow.ActiveRun,
    ):
        """Initialize recommendation system.

        Args:
            uidata: UserItemData with train/test split
            model: HybridMF model instance
            optimizer: Optimizer constructor (partial function)
            loss: Pairwise loss function
            device: Device for computation
            mlflow_run: Active MLflow run
        """
        print('Initiating Hybrid Recommender System..')

        # MLflow tracking
        self.mlflow_run = mlflow_run
        print(f'MLflow run ID: {self.mlflow_run.info.run_id}')

        # Device configuration
        self.device = (
            torch.device(device) if isinstance(device, str)
            else device
        )
        print(f'Using device: {self.device}')

        # Store data (already split into train/test)
        self.Rpos_train_csr = uidata.Rpos_train.tocsr()
        self.Rpos_test_csr = uidata.Rpos_test.tocsr()
        self.Rneg_train_csr = uidata.Rneg_train.tocsr()
        self.Rneg_test_csr = uidata.Rneg_test.tocsr()
        self.Fu_csr = uidata.Fu.tocsr()
        self.Fi_csr = uidata.Fi.tocsr()

        # All positives for negative sampling exclusion
        self.Rpos_all_csr = (
            uidata.Rpos_train + uidata.Rpos_test
        ).tocsr()

        # Model setup
        self.model = model.to(self.device)
        self.optimizer = optimizer(self.model.parameters())
        self.loss = loss

        # Valid users (present in both train and test)
        users_train = get_nonempty_rows_csr(uidata.Rpos_train)
        users_test = get_nonempty_rows_csr(uidata.Rpos_test)
        self.users = np.intersect1d(users_train, users_test)
        print(f'Got {len(self.users)} users for train/test')

    def _compute_auc_all_items(
        self,
        users: List[int],
        pos_csr_mat: csr_matrix,
        exclude_items_csr: Optional[csr_matrix] = None,
        max_items_limit: int = 1000000
    ) -> List[float]:
        """Compute AUC against all items for large catalogs.

        Args:
            users: User indices to evaluate
            pos_csr_mat: Positive items (label=1)
            exclude_items_csr: Items to exclude
            max_items_limit: Max items before raising error
        """
        n_items = self.Fi_csr.shape[0]

        # Safety check for large item catalogs
        if n_items > max_items_limit:
            raise ValueError(
                f"Too many items ({n_items:,}) for all-items "
                f"evaluation. Use eval_auc_neg_ratio > 0."
            )

        auc_scores = []

        for user_idx in users:
            # Get positive items
            pos_start = pos_csr_mat.indptr[user_idx]
            pos_end = pos_csr_mat.indptr[user_idx + 1]
            pos_items = pos_csr_mat.indices[pos_start:pos_end]

            if len(pos_items) == 0:
                continue

            # Get items to exclude
            if exclude_items_csr is not None:
                excl_start = exclude_items_csr.indptr[user_idx]
                excl_end = exclude_items_csr.indptr[user_idx + 1]
                excl_items = exclude_items_csr.indices[
                    excl_start:excl_end
                ]
            else:
                excl_items = np.array([], dtype=np.int32)

            # Create inclusion mask
            include_mask = np.ones(n_items, dtype=bool)
            include_mask[excl_items] = False
            included_items = np.where(include_mask)[0]

            if len(included_items) == 0:
                continue

            # Predict scores for all included items
            y_pred = self.model.predict(
                user_features=self.Fu_csr[[user_idx], :],
                item_features=self.Fi_csr[included_items, :]
            ).detach().numpy()[0, :]

            # Create binary labels
            y_true = np.isin(
                included_items, pos_items
            ).astype(np.int8)

            # Compute AUC
            user_auc = roc_auc_score(y_true, y_pred)
            auc_scores.append(user_auc)

        return auc_scores

    def _get_pos_neg_scores(
        self,
        users: List[int],
        pos_csr: csr_matrix,
        neg_csr: csr_matrix,
        exclude_pos_csr: Optional[csr_matrix] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _train(self, batch_users: np.ndarray) -> float:
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
        loss = self.loss(r_ui, r_uj)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_metrics(
        self,
        users: List[int],
        eval_auc_neg_ratio: float = 100.0,
    ) -> Dict[str, float]:
        """Evaluate AUC score and loss on test set."""
        metrics = {}
        with torch.no_grad():
            # Compute AUC
            if eval_auc_neg_ratio < 0:
                # All items mode (only for small catalogs)
                auc_scores = self._compute_auc_all_items(
                    users=users,
                    pos_csr_mat=self.Rpos_test_csr,
                    exclude_items_csr=self.Rpos_train_csr
                )
            else:
                # Sampled negatives mode (default)
                user_interactions = get_user_interactions(
                    users=users,
                    pos_csr_mat=self.Rpos_test_csr,
                    neg_csr_mat=self.Rneg_test_csr,
                    neg_ratio=eval_auc_neg_ratio,
                    exclude_pos_csr=self.Rpos_all_csr,
                )
                # Create predict function for AUC computation
                def predict_fn(user_idx, items):
                    return self.model.predict(
                        user_features=self.Fu_csr[[user_idx], :],
                        item_features=self.Fi_csr[items, :]
                    ).detach().numpy()[0, :]

                auc_scores = compute_auc_scores(
                    user_interactions=user_interactions,
                    predict_fn=predict_fn
                )

            metrics['auc'] = np.nanmean(auc_scores).item()
            metrics['auc_std'] = np.nanstd(auc_scores).item()

            # Compute loss
            r_ui, r_uj = self._get_pos_neg_scores(
                users,
                self.Rpos_test_csr,
                self.Rneg_test_csr,
                exclude_pos_csr=self.Rpos_all_csr,
            )
            loss = self.loss(r_ui, r_uj)
            metrics['loss'] = loss.item()

        return metrics

    def evaluate(
        self,
        max_users: Optional[int] = None,
        eval_auc_neg_ratio: float = 100.0,
    ) -> Dict[str, float]:
        """Evaluate model on test set."""
        self.model.eval()
        with torch.no_grad():
            # Select users for evaluation
            if max_users is None or max_users >= len(self.users):
                users = self.users.tolist()
            else:
                users = np.random.choice(
                    self.users,
                    size=max_users,
                    replace=False
                ).tolist()

            # Test metrics
            metrics = self.compute_metrics(
                users, eval_auc_neg_ratio=eval_auc_neg_ratio
            )

        self.model.train()
        return metrics

    def fit(
        self,
        n_iter: int,
        batch_size: int = 1000,
        eval_every: int = 5,
        eval_user_size: Optional[int] = None,
        early_stopping_patience: int = 10,
        eval_auc_neg_ratio: float = 100.0,
    ) -> None:
        """Train the model using BPR optimization."""
        # Detect subprocess for multiprocessing
        from pathos.helpers import mp
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

        # Log hyperparameters
        mlflow.log_params({
            'n_iter': n_iter,
            'batch_size': batch_size,
            'eval_every': eval_every,
            'eval_user_size': eval_user_size,
            'early_stopping_patience': early_stopping_patience,
            'eval_auc_neg_ratio': eval_auc_neg_ratio,
        })

        # Early stopping setup
        best_test_auc = 0.0
        best_epoch = 0
        patience_counter = 0

        # Progress tracking
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

            # Track average loss
            avg_loss = epoch_loss / n_batches
            epoch_looper.set_postfix({'loss': f'{avg_loss:.4f}'})

            # Periodic evaluation
            if epoch % eval_every == 0:
                eval_metrics = self.evaluate(
                    max_users=eval_user_size,
                    eval_auc_neg_ratio=eval_auc_neg_ratio,
                )
                current_auc = eval_metrics.get('auc', 0)

                # Log metrics to MLflow
                mlflow.log_metrics(eval_metrics, step=epoch)

                # Track best model
                if current_auc > best_test_auc:
                    best_test_auc = current_auc
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping check
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

                # Update progress bar
                epoch_looper.write(
                    f"E{epoch}: "
                    f"AUC {eval_metrics.get('auc', 0):.3f} | "
                    f"Loss {eval_metrics.get('loss', 0):.3f}"
                )

        # Save best model
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
