from __future__ import annotations

import urllib.error
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ========= 便利ユーティリティ =========


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_boston_from_cmu() -> tuple[np.ndarray, np.ndarray]:
    """
    Boston Housing を CMU 公開データから読み込む（scikit-learn docsの手順を参考に再実装）。
    """
    url = "http://lib.stat.cmu.edu/datasets/boston"
    raw = pd.read_csv(url, sep=r"\s+", skiprows=22, header=None)
    # 2行で1レコードになっているので左右連結してX、別列をyに
    X = np.concatenate([raw.values[::2, :], raw.values[1::2, :2]], axis=1)
    y = raw.values[1::2, 2]
    return X.astype(np.float32), y.astype(np.float32)


def load_california() -> tuple[np.ndarray, np.ndarray]:
    """
    推奨代替データ（California Housing）。
    """
    from sklearn.datasets import fetch_california_housing

    d = fetch_california_housing()
    X = d.data.astype(np.float32)
    y = d.target.astype(np.float32)
    return X, y


@dataclass
class DatasetConfig:
    use_boston_like: bool = True  # True: Boston(CMU) / False: California
    test_size: float = 0.2
    batch_size: int = 32
    shuffle: bool = True
    seed: int = 42


def make_loaders(cfg: DatasetConfig) -> tuple[DataLoader, DataLoader, int]:
    """
    データ読み込み→train/test分割→Tensor化→DataLoader化
    戻り値: train_loader, valid_loader, n_features
    """
    if cfg.use_boston_like:
        try:
            X, y = load_boston_from_cmu()
        except urllib.error.URLError as e:
            print("CMUへの接続に失敗。California Housingに切り替えます:", e)
            X, y = load_california()
    else:
        X, y = load_california()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed
    )
    # （元記事に合わせて）前処理は最小限。必要なら標準化を加えてください。
    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr).unsqueeze(1)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val).unsqueeze(1)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader, X.shape[1]


class MLPRegressor(nn.Module):
    """
    入力→64→64→出力(1) のシンプルな回帰MLP
    """

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_epoch(model, loader, criterion, optimizer, device: str = "cpu") -> float:
    model.train()
    running = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device: str = "cpu") -> float:
    model.eval()
    running = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        running += loss.item() * xb.size(0)
    return running / len(loader.dataset)


def plot_history(train_losses, val_losses):
    plt.figure(figsize=(4, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.title("Model loss")
    plt.ylabel("MSE")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def predict_all(model, loader, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb).cpu().numpy().ravel()
        preds.append(out)
        trues.append(yb.numpy().ravel())
    return np.concatenate(preds), np.concatenate(trues)


def plot_pred_vs_true(pred: np.ndarray, true: np.ndarray, title="Prediction vs. True"):
    plt.figure(figsize=(4, 4))
    plt.scatter(pred, true, s=12, alpha=0.7)
    plt.title(title)
    plt.ylabel("True value")
    plt.xlabel("Prediction")
    # 参考に対角線
    lo = min(pred.min(), true.min())
    hi = max(pred.max(), true.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.show()


# ========= メイン実行 =========


def main():
    # 1) 設定
    cfg = DatasetConfig(use_boston_like=True, test_size=0.2, batch_size=32, seed=42)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2) データ
    train_loader, val_loader, in_dim = make_loaders(cfg)

    # 3) モデル
    model = MLPRegressor(in_dim).to(device)
    print(model)

    # 4) 学習準備
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 5) 学習ループ（元記事に合わせて 100 エポック）
    n_epochs = 100
    train_hist, val_hist = [], []
    for epoch in range(1, n_epochs + 1):
        tl = train_epoch(model, train_loader, criterion, optimizer, device)
        vl = eval_epoch(model, val_loader, criterion, device)
        train_hist.append(tl)
        val_hist.append(vl)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: train MSE={tl:.3f}  val MSE={vl:.3f}")

    # 6) 学習曲線の可視化
    plot_history(train_hist, val_hist)

    # 7) 予測と可視化（検証データで実演）
    pred, true = predict_all(model, val_loader, device)
    plot_pred_vs_true(pred, true)


if __name__ == "__main__":
    main()
