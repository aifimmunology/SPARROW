import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
class Dataloader(Dataset):
    """
    PyTorch DataLoader for handling spatial data and optional scRNA-seq and labels

    Parameters:
        x: Tensor
            Spatial training data
        x_prime: Tensor, optional
            scRNA-seq training data
        label: Tensor, optional
            Label tensor (default: None).
    """
    def __init__(self, x, x_prime=None, label=None):
        self.x = x
        self.x_prime = x_prime
        self.label = label
        self.has_label = label is not None
        self.has_x_prime = x_prime is not None

    def __len__(self):
        return max(len(self.x), len(self.label)) if self.has_label else len(self.x)

    def __getitem__(self, idx):
        x_idx = idx % len(self.x)  # Cycle through x if necessary
        x = self.x[x_idx]
        label = self.label[idx % len(self.label)] if self.has_label else None
        x_prime = self.x_prime[idx % len(self.x_prime)] if self.has_x_prime else None
        return tuple(filter(lambda val: val is not None, (x, x_prime, label)))


class BalancedSampler(Sampler):
    """
    A custom sampler that samples underrepresented classes more frequently.

    Parameters:
    label: Tensor
        The tensor containing labels for balancing.
    """
    def __init__(self, label):
        self.label = label
        self.indices = self._create_balanced_indices()

    def _create_balanced_indices(self):
        label_indices = {label.item(): np.where(self.label.numpy() == label)[0] for label in np.unique(self.label.numpy())}
        max_samples = max(len(indices) for indices in label_indices.values())

        balanced_indices = []
        for indices in label_indices.values():
            repeated_indices = np.tile(indices, max_samples // len(indices))
            extra_indices = np.random.choice(indices, max_samples % len(indices), replace=False)
            combined_indices = np.concatenate((repeated_indices, extra_indices))
            balanced_indices.extend(combined_indices)

        np.random.shuffle(balanced_indices)
        return balanced_indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)