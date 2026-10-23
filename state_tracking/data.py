import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from state_tracking.groups import GROUPS


class StateTrackingDataset(Dataset):
    def __init__(self, group, num_examples, seq_len, seed=42):
        self.group = group
        self.num_examples = num_examples
        self.seq_len = seq_len

        rng = np.random.RandomState(seed)
        inputs = rng.randint(0, group.order, size=(num_examples, seq_len))
        labels = np.empty_like(inputs)
        for i in range(num_examples):
            labels[i] = group.cumulative_product(inputs[i])

        self.inputs = torch.from_numpy(inputs).long()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


def get_state_tracking_dataloaders(group_name, seq_len, train_examples=10000,
                                   test_examples=2000, batch_size=64, seed=42,
                                   num_workers=0):
    """(train_loader, test_loader, group). The test set uses seed + 1000."""
    group = GROUPS[group_name]()
    train_ds = StateTrackingDataset(group, train_examples, seq_len, seed=seed)
    test_ds = StateTrackingDataset(group, test_examples, seq_len, seed=seed + 1000)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, group
