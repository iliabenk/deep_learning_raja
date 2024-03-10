import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils
import numpy as np
from torchvision.datasets import FashionMNIST

from utils import mnist_reader, seed_handler
from configs import INIT_SEED
from configs import DATA_DIR, DATA_TYPE
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import datasets

class SVM_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MNIST_Dataset(Dataset):
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


def load_data(batch_size, num_labels, transform_func=None):
    seed_handler._set_seed(INIT_SEED) # To verify both SVM & VAE break the data in the same way

    if DATA_TYPE == "fashion":
        X_train, y_train = mnist_reader.load_mnist(DATA_TYPE, kind='train')
        X_test, y_test = mnist_reader.load_mnist(DATA_TYPE, kind='t10k')
    else:
        import torch
        from torchvision import datasets, transforms

        # Define a transform to normalize the data
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to PyTorch tensor
            # transforms.Normalize((0.5,), (0.5,))
        ])

        # Download and load the training data
        trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True)

        train_data, train_labels = next(iter(trainloader))
        X_train = train_data.numpy()
        y_train = train_labels.numpy()

        # Download and load the test data
        testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

        test_data, test_labels = next(iter(testloader))
        X_test = test_data.numpy()
        y_test = test_labels.numpy()

    shuffled_indices = np.arange(X_train.shape[0])
    np.random.shuffle(shuffled_indices)

    X_train_labeled = X_train[shuffled_indices[:num_labels], ...]
    y_train_labeled = y_train[shuffled_indices[:num_labels], ...]

    X_train_unlabeled = X_train[shuffled_indices[num_labels:], ...]
    y_train_unlabeled = X_train[shuffled_indices[num_labels:], ...]

    if transform_func:
        transform = transform_func()
    else:
        transform = None

    latent_train_loader = DataLoader(dataset=MNIST_Dataset(X_train_unlabeled, y_train_unlabeled, transform=transform),
                                     batch_size=batch_size,
                                     shuffle=True)

    classifier_train_loader = DataLoader(dataset=MNIST_Dataset(X_train_labeled, y_train_labeled, transform=transform),
                                         batch_size=batch_size,
                                         shuffle=True)

    test_loader = DataLoader(dataset=MNIST_Dataset(X_test, y_test, transform=transform),
                              batch_size=batch_size,
                              shuffle=False)

    return latent_train_loader, classifier_train_loader, test_loader

def normalize(x):
    mu = 0.2860 # Calculated on X_train
    sigma = 0.3530 # Calculated on X_train

    x = transforms.Normalize(mean=mu, std=sigma)(x)

    return x

def transform_func():
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Lambda(normalize),
        nn.Flatten(),
        transforms.Lambda(lambda x: torch.squeeze(x))
    ])

    return transform

def extract_features(vae_model, data_loader, device):
    vae_model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)

            mean, _ = vae_model.encoder(data)
            features.extend(mean.cpu().numpy())

            # mean, logvar = vae_model.encoder(data)
            # std = torch.exp(0.5 * logvar)
            # features.extend(torch.cat((mean, logvar), dim=1))
            # features.extend(sig.cpu().numpy())

            labels.extend(label.numpy())

    return np.array(features), np.array(labels)

#Q4

def get_data_loader(args):

    trans = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])
    train_dataset = FashionMNIST(root=args.dataroot, train=True, download=args.download, transform=trans)
    test_dataset = FashionMNIST(root=args.dataroot, train=False, download=args.download, transform=trans)


    # Check if everything is ok with loading datasets
    assert train_dataset
    assert test_dataset

    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = data_utils.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader


class gan_type:
    def __init__(self, model, is_train, download, dataroot, dataset, epochs, cuda, batch_size):
        self.model = model
        self.is_train = is_train
        self.download = download
        self.dataroot = dataroot
        self.dataset = dataset
        self.epochs = epochs
        self.cuda = cuda
        self.batch_size = batch_size
        self.load_D = 'discriminator.pkl'
        self.load_G = 'generator.pkl'
        self.channels = 1


def get_inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """
        Computes the inception score of the generated images imgs
        imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
        cuda -- whether or not to run on GPU
        batch_size -- batch size for feeding into Inception v3
        splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)