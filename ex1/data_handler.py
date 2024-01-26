from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import mnist_reader
from configs import DATA_DIR
import torchvision.transforms as transforms

class Lenet5_Dataset(Dataset):
    def __init__(self, X, y, transform=None):
        assert len(y) == X.shape[0], (f"len(y)={len(y)}, X.shape={X.shape}, which do not match.\n"
                                      f"Must maintain len(y) == X.shape[0]")

        self.X = np.reshape(X, (-1, 28, 28, 1))
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


def load_data(batch_size, transform_func=None):
    X_train, y_train = mnist_reader.load_mnist(DATA_DIR, kind='train')
    X_test, y_test = mnist_reader.load_mnist(DATA_DIR, kind='t10k')

    if transform_func:
        transform = transform_func()
    else:
        transform = None

    train_loader = DataLoader(dataset=Lenet5_Dataset(X_train, y_train, transform=transform),
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=Lenet5_Dataset(X_test, y_test, transform=transform),
                              batch_size=batch_size,
                              shuffle=False)

    return train_loader, test_loader

def normalize(x):
    mu = 0.2860 # Calculated on X_train
    sigma = 0.3530 # Calculated on X_train

    x = transforms.Normalize(mean=mu, std=sigma)(x)

    return x

def transform_func():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(normalize)
    ])

    return transform
