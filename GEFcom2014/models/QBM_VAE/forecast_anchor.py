# -*- coding: utf-8 -*-

"""Validation-selected deterministic anchors for residual QBM decoders."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DeterministicForecastAnchor(nn.Module):
    def __init__(self, context_dim, target_dim, hidden_dim=256, layers=2):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        widths = [context_dim] + [hidden_dim] * layers + [target_dim]
        modules = []
        for index, (in_features, out_features) in enumerate(zip(widths[:-1], widths[1:])):
            modules.append(nn.Linear(in_features, out_features))
            if index < len(widths) - 2:
                modules.append(nn.SiLU())
        self.network = nn.Sequential(*modules)

    def forward(self, context):
        return self.network(context)


def fit_deterministic_anchor(
    x_ls,
    y_ls,
    x_vs,
    y_vs,
    hidden_dim=256,
    layers=2,
    epochs=100,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=1e-5,
    patience=15,
    min_delta=1e-5,
    seed=0,
    device=None,
):
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")
    device = device or torch.device("cpu")
    model = DeterministicForecastAnchor(
        context_dim=x_ls.shape[1],
        target_dim=y_ls.shape[1],
        hidden_dim=hidden_dim,
        layers=layers,
    ).to(device)
    dataset = TensorDataset(
        torch.as_tensor(x_ls, dtype=torch.float32),
        torch.as_tensor(y_ls, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    x_vs_tensor = torch.as_tensor(x_vs, dtype=torch.float32, device=device)
    y_vs_tensor = torch.as_tensor(y_vs, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state = None
    best_validation = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(epochs):
        model.train()
        train_sum = 0.0
        train_count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_sum += float(loss.detach()) * batch_x.shape[0]
            train_count += batch_x.shape[0]

        model.eval()
        with torch.no_grad():
            validation_loss = float(
                torch.nn.functional.mse_loss(model(x_vs_tensor), y_vs_tensor)
            )
        train_loss = train_sum / train_count
        history.append(
            {
                "epoch": int(epoch),
                "train_mse": float(train_loss),
                "validation_mse": validation_loss,
            }
        )
        if validation_loss < best_validation - min_delta:
            best_validation = validation_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                "Anchor epoch %d | LS MSE %.6f VS MSE %.6f"
                % (epoch + 1, train_loss, validation_loss)
            )
        if stale_epochs >= patience:
            print("Anchor early stopping at epoch %d" % (epoch + 1))
            break

    if best_state is None:
        raise RuntimeError("Deterministic anchor did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_deterministic_anchor(model, context, device=None, batch_size=1024):
    device = device or next(model.parameters()).device
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, context.shape[0], batch_size):
            batch = torch.as_tensor(
                context[start:start + batch_size], dtype=torch.float32, device=device
            )
            chunks.append(model(batch).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)
