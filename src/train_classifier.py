"""ResNet18 ile NEU-DET hata sınıflandırması (transfer learning)."""

import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torchvision import models

from dataset import get_loaders
from voc_to_yolo import CLASSES

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

EPOCHS = 15
LR = 1e-4
BATCH_SIZE = 32
SEED = 42


def get_device():
    """Varsa Mac GPU'sunu (MPS) kullan, yoksa CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(num_classes):
    """ImageNet'te eğitilmiş ResNet18'i al, son katmanı bizim sınıf sayımıza göre değiştir."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Bir tur çalıştır. optimizer verilirse eğitir, verilmezse sadece ölçer."""
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            count += images.size(0)

    return total_loss / count, correct / count


def main():
    torch.manual_seed(SEED)
    device = get_device()
    print("Cihaz:", device)

    train_loader, val_loader, _ = get_loaders(BATCH_SIZE)
    model = build_model(len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    MODEL_DIR.mkdir(exist_ok=True)
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })
        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"train loss {train_loss:.3f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.3f} acc {val_acc:.3f} | "
              f"{time.time() - start:.0f} sn")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_DIR / "resnet18_best.pt")
            print("   -> en iyi model kaydedildi")

    pd.DataFrame(history).to_csv(REPORT_DIR / "classifier_history.csv", index=False)
    print("Bitti. En düşük val loss:", round(best_val_loss, 4))


if __name__ == "__main__":
    main()