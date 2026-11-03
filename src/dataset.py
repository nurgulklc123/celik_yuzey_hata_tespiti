"""NEU-DET sınıflandırma için PyTorch Dataset ve DataLoader."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

from voc_to_yolo import CLASSES

PROJECT_DIR = Path(__file__).resolve().parent.parent
IMG_DIR = PROJECT_DIR / "data" / "NEU-DET" / "IMAGES"
SPLIT_CSV = PROJECT_DIR / "splits" / "split.csv"

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(train):
    """Resmi modele hazırlayan adımlar. train=True ise veri artırma da eklenir."""
    steps = [
        T.Grayscale(num_output_channels=3),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]
    if train:
        steps += [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomChoice([
                T.RandomRotation((0, 0)),
                T.RandomRotation((90, 90)),
                T.RandomRotation((180, 180)),
                T.RandomRotation((270, 270)),
            ]),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    steps += [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(MEAN, STD),
    ]
    return T.Compose(steps)


class NEUDataset(Dataset):
    """split.csv'den bir grubun (train / val / test) resimlerini ve etiketlerini verir."""

    def __init__(self, split, transform=None):
        df = pd.read_csv(SPLIT_CSV)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(IMG_DIR / row["file"])
        label = CLASSES.index(row["class"])
        if self.transform:
            img = self.transform(img)
        return img, label


def get_loaders(batch_size=32):
    """Üç grup için DataLoader'ları hazırlar."""
    train_ds = NEUDataset("train", get_transforms(train=True))
    val_ds = NEUDataset("val", get_transforms(train=False))
    test_ds = NEUDataset("test", get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders()
    print("Train:", len(train_loader.dataset),
          "Val:", len(val_loader.dataset),
          "Test:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))
    print("Bir tepsinin şekli:", images.shape)
    print("İlk 10 etiket:", labels[:10].tolist())
    print("Piksel aralığı:", round(images.min().item(), 2), "ile", round(images.max().item(), 2))