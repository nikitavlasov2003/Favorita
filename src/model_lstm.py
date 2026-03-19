import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

class FavoritaDataset(Dataset):

    def __init__(self, df, seq_len=30, horizon=15):
        self.seq_len = seq_len
        self.horizon = horizon

        # Ожидаемые колонки: log_unit_sales, onpromotion, dcoilwtico, is_holiday,
        # store_idx, family_idx, cluster_idx, perishable
        self.sales = df['log_unit_sales'].values.astype(np.float32)
        self.exog = df[["onpromotion", "dcoilwtico", "is_holiday"]].values.astype(np.float32)
        self.store_ids = df['store_idx'].values
        self.family_ids = df['family_idx'].values
        self.cluster_ids = df['cluster_idx'].values
        self.perishable = df['perishable'].values.astype(np.float32)

        # Определяем границы серий (смена товара или магазина)
        series_change = (df['item_nbr'].diff() != 0) | (df['store_nbr'].diff() != 0)
        series_starts = np.where(series_change)[0]
        if len(series_starts) == 0:
            series_starts = np.array([0])
        series_ends = np.append(series_starts[1:], len(df))

        self.valid_indices = []
        for start, end in zip(series_starts, series_ends):
            if end - start >= (seq_len + horizon):
                self.valid_indices.extend(range(start, end - seq_len - horizon + 1))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]

        # Encoder (прошлое)
        s_begin = start_idx
        s_end = start_idx + self.seq_len
        x_enc = np.concatenate([
            self.sales[s_begin:s_end, None],
            self.exog[s_begin:s_end]
        ], axis=1)

        # Decoder (будущие экзогенные факторы)
        d_begin = s_end
        d_end = s_end + self.horizon
        x_dec = self.exog[d_begin:d_end]

        # Target
        y = self.sales[d_begin:d_end]

        return {
            "x_enc": torch.tensor(x_enc, dtype=torch.float32),
            "x_dec": torch.tensor(x_dec, dtype=torch.float32),
            "store_id": torch.tensor(self.store_ids[s_begin], dtype=torch.long),
            "family_id": torch.tensor(self.family_ids[s_begin], dtype=torch.long),
            "cluster_id": torch.tensor(self.cluster_ids[s_begin], dtype=torch.long),
            "perishable": torch.tensor(self.perishable[s_begin], dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32)
        }


class LSTMForecaster(nn.Module):
    """LSTM encoder–decoder с эмбеддингами."""

    def __init__(
        self,
        hidden_size=64,
        num_layers=1,
        dropout=0.2,
        horizon=15,
        n_stores=60,
        n_families=35,
        n_clusters=25,
        embed_dim=8,
        n_exog=3  # onpromotion, dcoilwtico, is_holiday
    ):
        super().__init__()
        self.horizon = horizon

        self.store_emb = nn.Embedding(n_stores + 1, embed_dim)
        self.family_emb = nn.Embedding(n_families + 1, embed_dim)
        self.cluster_emb = nn.Embedding(n_clusters + 1, embed_dim)

        self.lstm = nn.LSTM(
            input_size=1 + n_exog,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.bn = nn.BatchNorm1d(hidden_size)

        dec_input_size = hidden_size + 3 * embed_dim + 1 + n_exog * horizon
        self.decoder = nn.Sequential(
            nn.Linear(dec_input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, horizon)
        )

    def forward(self, batch):
        x_enc = batch["x_enc"]  # (B, seq_len, 1+n_exog)
        x_dec = batch["x_dec"]  # (B, horizon, n_exog)

        _, (h_n, _) = self.lstm(x_enc)
        h = h_n[-1]
        h = self.bn(h)

        e_store = self.store_emb(batch["store_id"])
        e_family = self.family_emb(batch["family_id"])
        e_cluster = self.cluster_emb(batch["cluster_id"])
        perish = batch["perishable"].unsqueeze(1)

        x_dec_flat = x_dec.reshape(x_dec.size(0), -1)
        combined = torch.cat([h, e_store, e_family, e_cluster, perish, x_dec_flat], dim=1)

        return self.decoder(combined)


def train_lstm(
    train_df: pd.DataFrame,
    params: dict = None,
    device: torch.device = None,
    verbose: bool = True
) -> nn.Module:
    """
    Train LSTM model on prepared data.

    Args:
        train_df: pandas DataFrame, должен быть отсортирован по (store_nbr, item_nbr, date)
                  и содержать колонки:
                  - log_unit_sales
                  - onpromotion, dcoilwtico, is_holiday
                  - store_idx, family_idx, cluster_idx (индексы для эмбеддингов)
                  - perishable
                  - store_nbr, item_nbr (для определения границ серий)
        params: словарь с параметрами (по умолчанию указаны ниже)
        device: устройство для обучения (автоопределение, если None)
        verbose: печатать прогресс

    Returns:
        обученная модель
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_params = {
        "seq_len": 30,
        "horizon": 15,
        "batch_size": 1024,
        "hidden_size": 64,
        "num_layers": 1,
        "lr": 0.001,
        "epochs": 3,
        "dropout": 0.2,
        "clip_grad": 1.0,
        "embed_dim": 8,
        "n_exog": 3  # onpromotion, dcoilwtico, is_holiday
    }
    if params:
        default_params.update(params)
    p = default_params

    # Датасет и загрузчик
    dataset = FavoritaDataset(train_df, seq_len=p["seq_len"], horizon=p["horizon"])
    loader = DataLoader(
        dataset,
        batch_size=p["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Размерности эмбеддингов
    n_stores = train_df['store_idx'].nunique()
    n_families = train_df['family_idx'].nunique()
    n_clusters = train_df['cluster_idx'].nunique()

    model = LSTMForecaster(
        hidden_size=p["hidden_size"],
        num_layers=p["num_layers"],
        dropout=p["dropout"],
        horizon=p["horizon"],
        n_stores=n_stores,
        n_families=n_families,
        n_clusters=n_clusters,
        embed_dim=p["embed_dim"],
        n_exog=p["n_exog"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=p["lr"])
    criterion = nn.MSELoss()

    if verbose:
        print(f"Training on {device} | {len(dataset)} windows | {len(loader)} batches per epoch")

    for epoch in range(1, p["epochs"] + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{p['epochs']}", disable=not verbose)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(batch)
            loss = criterion(preds, batch["y"])

            optimizer.zero_grad()
            loss.backward()
            if p["clip_grad"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), p["clip_grad"])
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(loader)
        if verbose:
            print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")

    return model
