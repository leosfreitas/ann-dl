"""MNIST loading and the fixed 50 000 / 10 000 split (Exercise 1A).

Kept in one place so Exercise 1 and Exercise 2 provably train on the same data:
the split generator is seeded with 42, exactly as the statement prescribes.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATA_DIR = "data"  # relative to this folder; MNIST downloads here once


def load_mnist(batch_size=128, data_dir=DATA_DIR):
    """Return (train_loader, val_loader, test_loader, test_set).

    Pixels stay in [0, 1] -- ``ToTensor()`` already does that -- and are
    flattened to 784-vectors by the training loop, not here, so the raw
    (1, 28, 28) tensors remain available for plotting.
    """
    train_full = datasets.MNIST(data_dir, train=True, download=True,
                                transform=transforms.ToTensor())
    test = datasets.MNIST(data_dir, train=False, download=True,
                          transform=transforms.ToTensor())

    # The split prescribed by the statement, with its own seeded generator.
    train, val = random_split(train_full, [50_000, 10_000],
                              generator=torch.Generator().manual_seed(42))

    # shuffle=True only on the training loader; the two evaluation loaders are
    # deterministic, and num_workers=0 keeps the whole run reproducible.
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val, batch_size=512, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=512, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, test


def test_matrix(test_set):
    """The 10 000 test images as one (10000, 784) tensor, plus their labels."""
    x = test_set.data.float().div(255.0).view(-1, 784)
    y = test_set.targets.clone()
    return x, y


def first_index_per_digit(test_set):
    """Index of the first test image of each digit 0-9 (Figure 2)."""
    y = test_set.targets
    return [int((y == digit).nonzero()[0].item()) for digit in range(10)]
