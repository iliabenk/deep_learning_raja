import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
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

############## Q4 ###################


# Initialize the weights of the networks
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Define the Wasserstein loss function
def wasserstein_loss(output, target):
    return torch.mean(output * target)


def load_Fashionmnist(batch_size=128, architecture='WGAN'):

    if architecture == 'WGAN':
      transform = transforms.Compose([
          transforms.Resize(32),
          transforms.ToTensor()])
    else:
      transform = transforms.Compose([transforms.ToTensor()])

    # set data FashionMnist
    train_data = datasets.FashionMNIST('../fashion_data', train=True, download=True,
                                       transform=transform)
    test_data = datasets.FashionMNIST('../fashion_data', train=False,
                                      transform=transform)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # # Plot some training images
    # real_batch = next(iter(train_loader))
    # plt.figure(figsize=(8, 8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(
    #     np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    # plt.show()

    return train_loader, test_loader


# Function to calculate gradient penalty for WGAN
def calculate_gradient_penalty(discriminator, real_images, fake_images):
    epsilon = torch.rand(len(real_images), 1, 1, 1).to(device)
    interpolated = epsilon * real_images + (1 - epsilon) * fake_images
    interpolated.requires_grad_(True)
    prob_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


