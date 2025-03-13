"""DataModule for exclusive-OR (XOR) dataset."""

# Imports PyTorch packages
import torch

# Imports milkshake packages.
from milkshake.datamodules.datamodule import DataModule
from milkshake.datamodules.dataset import Dataset


class XORDataset(Dataset):
    """Dataset for exclusive-OR (XOR)."""

    def __init__(self, *xargs, **kwargs):
        Dataset.__init__(self, *xargs, **kwargs)

    def download(self):
        pass

    def generate_data(self, gen):
        # 200 datapoints of dimension 10 each on the unit hypercube.
        self.data = (torch.randint(0, 2, (200, 10), generator=gen) * 2 - 1).float()
        
        # XOR function on first two dimensions.
        self.targets = -(self.data[:, 0] * self.data[:, 1])
        self.targets = torch.signbit(self.targets).int() # 0 or 1

    def load_data(self):
        seed = 0 if self.train else 1
        gen = torch.Generator().manual_seed(seed)
        self.generate_data(gen)

class XOR(DataModule):
    """DataModule for exclusive-OR (XOR)."""

    def __init__(self, args, **kwargs):
        super().__init__(args, XORDataset, 2, 0, **kwargs)
        self.shuffle = False
