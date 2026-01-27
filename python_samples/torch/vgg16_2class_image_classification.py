from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm, trange  # 追加


# ====== 1) データ読み込み：CIFAR-10のクラス0/1のみ抽出 ======
def get_cifar10_two_classes(
    root: str = "./data",
    classes_keep: List[int] = [0, 1],
) -> Tuple[Subset, Subset]:
    """
    CIFAR-10 からクラス0/1のみを抽出した train/test の Subset を返します。
    Keras記事の get_input() 相当。 参考: 元記事の「class=0,1のみ抽出」.
    """
    # Keras記事では 1/255 のスケーリング＋回転/平行移動/水平反転を学習側のみ付与
    # PyTorch版でもそれに合わせます（Normalizeはオフ。必要なら後述のNOTE参照）。
    train_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),  # 念のため固定
            transforms.RandomRotation(degrees=90),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),  # [0,1]スケーリング
            # NOTE: ここでNormalizeを入れるとImageNet学習済みと相性◎（後述）
            # transforms.Normalize(mean=[0.485,0.456,0.406],
            #                      std=[0.229,0.224,0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    trainset = datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_transform
    )
    testset = datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_transform
    )

    # クラス0/1のみのインデックスを抽出
    def filter_indices(ds: datasets.CIFAR10, keep: List[int]) -> List[int]:
        # datasets.CIFAR10.targets は list[int]
        return [i for i, t in enumerate(ds.targets) if t in keep]

    train_idx = filter_indices(trainset, classes_keep)
    test_idx = filter_indices(testset, classes_keep)

    return Subset(trainset, train_idx), Subset(testset, test_idx)


# ====== 2) モデル定義：VGG16の畳み込み部を凍結し、ヘッドだけ学習 ======
class VGG16BinaryClassifier(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        freeze_features: bool = True,
        hidden_dim: int = 128,  # ← 追加
        dropout: float = 0.5,  # ← 追加
    ):
        super().__init__()
        base = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features = base.features
        if freeze_features:
            for p in self.features.parameters():
                p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


# ====== 3) 学習・評価ループ ======
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    criterion = nn.BCEWithLogitsLoss()
    for images, targets in loader:
        images = images.to(device)
        # CIFAR-10のラベルは0/1のままでOK。floatにして BCE 損失に合わせる
        targets = targets.float().unsqueeze(1).to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == targets.long()).sum().item()
        total += targets.size(0)
        total_loss += loss.item() * targets.size(0)
    return total_loss / total, correct / total


def train(epochs=5, batch_size=32, lr=1e-3, root="./data", num_workers=2, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_ds = get_cifar10_two_classes(root=root)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
    )

    model = VGG16BinaryClassifier(pretrained=True, freeze_features=True).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.BCEWithLogitsLoss()

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # 外側：エポックのプログレスバー（trangeはtqdm(range)の糖衣）
    for epoch in trange(1, epochs + 1, desc="Epochs"):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        # 内側：バッチのプログレスバー
        with tqdm(
            train_loader,
            desc=f"Train {epoch}/{epochs}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        ) as pbar:
            for images, targets in pbar:
                images = images.to(device)
                targets = targets.float().unsqueeze(1).to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                preds = (torch.sigmoid(logits) >= 0.5).long()
                running_correct += (preds == targets.long()).sum().item()
                running_total += targets.size(0)
                running_loss += loss.item() * targets.size(0)

                # 途中経過をバー右側に表示
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{(running_correct / running_total):.3f}",
                    }
                )

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_loss, val_acc = evaluate(model, test_loader, device)
        hist["train_loss"].append(train_loss)
        hist["train_acc"].append(train_acc)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)

        # エポック終了時の要約はprintでも残す
        print(
            f"Epoch {epoch}/{epochs} - "
            f"loss: {train_loss:.4f} acc: {train_acc:.4f} "
            f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}"
        )

    return model, hist


def plot_history(hist):
    # Accuracy
    plt.figure(figsize=(4, 5))
    plt.plot(hist["train_acc"], label="Train")
    plt.plot(hist["val_acc"], label="Test")
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.show()

    # Loss
    plt.figure(figsize=(4, 5))
    plt.plot(hist["train_loss"], label="Train")
    plt.plot(hist["val_loss"], label="Test")
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    # 元記事と同じくエポック5・バッチ32をデフォルトに
    model, hist = train(epochs=5, batch_size=32, lr=1e-3)
    plot_history(hist)
