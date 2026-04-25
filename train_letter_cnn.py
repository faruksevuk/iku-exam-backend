"""
Train a letter CNN on EMNIST 'letters' split (26 classes A-Z).

This script trains the same _MiniLetterCNN architecture used by handwriting.py,
applying the EMNIST orientation fix during training so the model expects
"human readable" orientation at inference time. Saves to models/letter_cnn.pt.

Run:
    python train_letter_cnn.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MiniLetterCNN(nn.Module):
    def __init__(self, num_classes: int = 26):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _emnist_orient_fix(t: torch.Tensor) -> torch.Tensor:
    """
    Convert raw EMNIST tensor (1, 28, 28) to human-readable orientation.
    EMNIST raw is rotated 90 CCW + flipped horizontally; the fix is
    transpose + flip along width axis.
    """
    return t.transpose(1, 2).flip(2)


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(base, "data")
    save_path = os.path.join(base, "models", "letter_cnn.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Standard MNIST normalization (works well for EMNIST too)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(_emnist_orient_fix),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    print("Loading EMNIST 'letters' split (will use existing data/EMNIST/raw if available)...")
    trainset = datasets.EMNIST(
        root=data_root, split="letters", train=True, download=True, transform=transform,
    )
    testset = datasets.EMNIST(
        root=data_root, split="letters", train=False, download=True, transform=transform,
    )
    print(f"  train: {len(trainset)} samples")
    print(f"  test:  {len(testset)} samples")
    # EMNIST 'letters' labels are 1-indexed (1-26 for A-Z) → shift to 0-25 in loop

    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=0)

    model = MiniLetterCNN(num_classes=26)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 5
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            y = y - 1  # EMNIST letters labels: 1..26 -> 0..25
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum())
            total += x.size(0)
        train_acc = correct / total

        # eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                y = y - 1
                preds = model(x).argmax(dim=1)
                correct += int((preds == y).sum())
                total += x.size(0)
        test_acc = correct / total

        print(f"epoch {epoch+1}/{EPOCHS} loss={running/len(trainset):.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
