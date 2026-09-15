"""
RiskFecta Phase 5A — pooled LSTM regression model (ML_SPEC.md §9, §17, §18;
BUILD_PLAN.md Phase 5).

Architecture, training-loop, and inference plumbing only. No sequence
construction (`pipeline.sequences`), no scaler fitting (`pipeline.scaling`),
no fold/training-set selection (`pipeline.folds` — and see
`pipeline.sequences`'s still-open TRAIN_WINDOW/LSTM_SEQ decision gate), no
metrics (`models.metrics`) — mirrors `models/xgboost_model.py`'s separation
of concerns exactly (task brief §11's "keep training code separate from
fold construction and metrics" applied to the LSTM branch too).

HARD GATE (task brief §2, §21): nothing in this module is ever called
against real Bloomberg-derived feature/sequence data in Phase 5A. Every
test exercises it only against small synthetic fixtures (task brief §20-P)
to prove the architecture, training loop, and reproducibility plumbing are
correct BEFORE real execution is authorized.

================================================================================
NON-FROZEN HYPERPARAMETERS — flagged for approval (task brief §13, §26)
================================================================================
Neither ML_SPEC.md nor BUILD_PLAN.md Phase 5 freezes LSTM hidden size,
layer count, dropout, optimizer, learning rate, batch size, or the
epoch/early-stopping policy. Per task brief §13 ("if parameters are not
frozen: do not conduct real-data OOS tuning ... use simple defensible
defaults only where the Build Plan permits, document them, and flag them
for approval before real execution"), the `DEFAULT_*` constants below are
simple, small, defensible MVP choices — never selected using any real OOS
result — and are NOT to be treated as final without explicit approval
before Phase 5B real training:

  - hidden_size=32, num_layers=1, dropout=0.0 (single layer -> torch's own
    `dropout` argument is inapplicable between layers; explicitly 0.0
    rather than silently ignored)
  - optimizer: Adam, lr=1e-3
  - batch_size=64
  - max_epochs=50 with early stopping (patience=5) on internal-validation
    MSE only (task brief §12 — outer OOS never seen by this function)
  - loss: MSE (regression), per ML_SPEC.md §15 "Expected MVP concept is
    regression loss, likely MSE"
  - seed=42 (matches `models.xgboost_model.DEFAULT_SEED`, for consistency
    across the two model branches — not independently meaningful)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from torch import nn

import config

# ---------------------------------------------------------------------------
# Locked feature surface (Flag 2, resolved — see pipeline.sequences module
# docstring for the full justification). Re-exported from config.py, the
# single source of truth, never independently redefined here.
# ---------------------------------------------------------------------------
LSTM_FEATURE_COLS = list(config.LSTM_ALL_FEATURES)
SEQUENCE_LENGTH = config.LSTM_SEQ

DEFAULT_HIDDEN_SIZE = 32
DEFAULT_NUM_LAYERS = 1
DEFAULT_DROPOUT = 0.0
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_EPOCHS = 50
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_SEED = 42

# Forbidden architecture components (task brief §13: "do NOT add attention/
# transformer/GRU branch/bidirectional LSTM/ticker embeddings/multi-task
# heads/custom exotic loss/portfolio loss/extra neural-network
# architectures"). Checked defensively in PooledLSTM.__init__.
_FORBIDDEN_MODULE_TYPES = (nn.MultiheadAttention, nn.TransformerEncoder, nn.GRU, nn.Embedding)


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Best-effort determinism across python/numpy/torch (task brief §16).
    Documented limitation: reproducible run-to-run on the same
    machine/torch build for the CPU-only ops used here; bit-for-bit
    determinism across different hardware/torch/BLAS versions is not
    guaranteed and is never claimed (task brief §16)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PooledLSTM(nn.Module):
    """One pooled LSTM (ML_SPEC.md §9; task brief §7) -> one scalar
    21-session forward-return regression prediction per input sequence
    (task brief §14: "Do not convert this into binary classification").

    Exactly one `nn.LSTM` stack (unidirectional, no ticker embedding, no
    attention) followed by one `nn.Linear` head reading the final layer's
    final-timestep hidden state — no architecture beyond what
    ML_SPEC.md/BUILD_PLAN.md's MVP concept implies (task brief §13).
    """

    def __init__(
        self,
        input_size: int = len(LSTM_FEATURE_COLS),
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)
        assert_no_forbidden_architecture(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_size) -> (batch,) scalar predictions."""
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size): final layer's final-timestep hidden state
        return self.head(last_hidden).squeeze(-1)


def assert_no_forbidden_architecture(model: nn.Module) -> None:
    """Defense in depth: fail loudly if a forbidden component (attention,
    transformer, GRU, embedding) is ever present on a PooledLSTM instance
    (task brief §13)."""
    for module in model.modules():
        if isinstance(module, _FORBIDDEN_MODULE_TYPES):
            raise ValueError(
                f"PooledLSTM: forbidden architecture component present: {type(module).__name__}"
            )


# ---------------------------------------------------------------------------
# Training — internal-temporal-validation-driven only; no outer-OOS access.
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    model: PooledLSTM
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    best_epoch: int = -1
    stopped_early: bool = False


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_layers: int = DEFAULT_NUM_LAYERS,
    dropout: float = DEFAULT_DROPOUT,
    lr: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    seed: int = DEFAULT_SEED,
) -> TrainResult:
    """Fit one PooledLSTM via MSE regression loss + Adam.

    Early stopping is driven ONLY by `(X_val, y_val)` — an INTERNAL temporal
    validation split the caller supplies (e.g.
    `pipeline.folds.internal_validation_split` /`rows_in_session_range`
    applied to a sequence index, task brief §12). This function has no
    parameter for, and never sees, an outer-OOS set: outer-OOS blindness is
    structural here, not merely conventional (task brief §12: "the outer
    OOS block must NEVER be used for any of these").

    Restores the best-validation-loss epoch's weights before returning
    (task brief §12 "the internal temporal validation set may drive [early
    stopping]").
    """
    set_seed(seed)
    input_size = X_train.shape[-1]
    model = PooledLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_train_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32)
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32)

    n = X_train_t.shape[0]
    if n == 0:
        raise ValueError("train_lstm: X_train has zero samples")

    result = TrainResult(model=model)
    best_val = float("inf")
    best_state = None
    epochs_since_improvement = 0
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=generator)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        result.train_losses.append(epoch_loss / n)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        result.val_losses.append(val_loss)

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            result.best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                result.stopped_early = True
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return result


def predict_lstm(model: PooledLSTM, X: np.ndarray) -> np.ndarray:
    """Predict for an already-scaled `(n, seq_len, n_features)` array.
    Returns a plain 1-D numpy array of scalar predictions (task brief §14) —
    caller is responsible for attaching ticker/date identity, exactly like
    `models.xgboost_model.predict_xgboost`."""
    model.eval()
    with torch.no_grad():
        X_t = torch.as_tensor(X, dtype=torch.float32)
        preds = model(X_t)
    return preds.numpy()
