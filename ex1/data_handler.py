from torch.utils.data import Dataset, DataLoader
import numpy as np

class Lenet5_Dataset(Dataset):
    def __init__(self, X, y, transform=None):
        assert len(y) == X.shape[0], (f"len(y)={len(y)}, X.shape={X.shape}, which do not match.\n"
                                      f"Must maintain len(y) == X.shape[0]")

        self.X = np.reshape(X, (-1, 1, 28, 28))
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
