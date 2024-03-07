import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import timeit
from torch.nn import functional as F
import random
from sklearn.svm import SVC
import torchvision.utils as vutils
import torch.optim as optim

from configs import LABELS, INIT_SEED, TENSORBOARD_DIR, MODELS_OUTPUT_DIR, IS_OPTIMIZE_LR, DATA_TYPE, \
    CIFAR10_ARCHITECTURE
from utils import seed_handler, model_utils
from model import VAE, DCGAN_Generator, DCGAN_Discriminator, WGAN_Generator, WGAN_Discriminator
from data_handler import load_data, transform_func, extract_features, weights_init, wasserstein_loss, load_Fashionmnist, \
    calculate_gradient_penalty

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def fit_vae(num_labels, is_save_tensorboard, params):
    lr = params['lr']
    latent_epochs = params['latent_epochs']
    classifier_epochs = params['classifier_epochs']
    batch_size = params['batch_size']
    optimizer_method = params['optimizer']
    lr_scheduler_enabled = params['lr_scheduler']['enable']
    lr_scheduler_gamma = params['lr_scheduler']['gamma']
    lr_scheduler_step = params['lr_scheduler']['step']

    if is_save_tensorboard:
        # Set tensorboard directory
        if lr_scheduler_enabled:
            tensorboard_dir = os.path.join(TENSORBOARD_DIR, f"labels_{num_labels}__latent_epochs_"
                                                            f"{latent_epochs}__classifier_epochs_{classifier_epochs}"
                                                            f"__lr_{lr}__batchSize"
                                                            f"_{batch_size}__optimizer_"
                                                            f"{optimizer_method}__lr_scheduler_params__enabled_"
                                                            f"{lr_scheduler_enabled}__gamma_{lr_scheduler_gamma}__step_"
                                                            f"{lr_scheduler_step}")

        else:
            tensorboard_dir = os.path.join(TENSORBOARD_DIR, f"labels_{num_labels}__latent_epochs_"
                                                            f"{latent_epochs}__classifier_epochs_{classifier_epochs}"
                                                            f"__lr_{lr}__batchSize"
                                                            f"_{batch_size}__optimizer_"
                                                            f"{optimizer_method}__lr_scheduler_params__enabled_"
                                                            f"{lr_scheduler_enabled}")

        # TensorBoard
        writer = SummaryWriter(log_dir=tensorboard_dir)

    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set Data
    latent_train_loader, classifier_train_loader, test_loader = load_data(batch_size, num_labels=100,
                                                                          transform_func=transform_func)

    # Model
    latent_space_model = VAE().to(device)

    # Optimizer
    if optimizer_method == 'adam':
        optimizer = torch.optim.Adam(latent_space_model.parameters(), lr=lr)
    elif optimizer_method == 'sgd':
        optimizer = torch.optim.SGD(latent_space_model.parameters(), lr=lr)
    else:
        assert False, f"optimizer={optimizer_method} is not supported"

    print(f"Optimizer = {optimizer_method}")

    scheduler = None
    if lr_scheduler_enabled:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step,
                                                    gamma=lr_scheduler_gamma)

    train_loss_history = []

    # VAE Train
    tic = timeit.default_timer()
    for epoch in range(latent_epochs):
        latent_space_model.train()
        epoch_correct = 0
        epoch_loss = 0
        for i, (data, _) in enumerate(latent_train_loader):
            data = data.float().to(device)

            # Forward
            reconstruction, mean, logvar = latent_space_model(data)

            # Reconstruction loss
            reconstruction_loss = F.binary_cross_entropy(reconstruction, data, reduction="sum")

            # KL divergence
            kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

            loss = (reconstruction_loss + kl_div) / batch_size

            # Total loss
            epoch_loss += loss.item()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % (len(latent_train_loader) // 10) == 0:
                toc = timeit.default_timer()
                print(f"VAE: Training: epoch={epoch}/{latent_epochs}, step={i}/{len(latent_train_loader)}, lr={lr}, "
                      f"labels={num_labels}, train_loss={loss.item()}, "
                      f"Training time so far: {round((toc - tic) / 60)} mins")

        if scheduler is not None:
            scheduler.step()

        epoch_loss /= len(latent_train_loader)
        train_loss_history.append(epoch_loss)

        if is_save_tensorboard:
            # Log metrics to TensorBoard
            writer.add_scalar('VAE Train/Loss', epoch_loss, epoch)

        print(f"Labels: {num_labels}, Train: epoch [{epoch + 1}/{latent_epochs}], Loss = {epoch_loss:.6f}")

    if is_save_tensorboard:
        writer.close()

    return latent_space_model, train_loss_history[-1]


def fit_svm(num_labels, kernel, vae_model):
    # Choose device
    device = 'cpu'
    vae_model.to(device)

    # Set Data
    _, classifier_train_loader, test_loader = load_data(1, num_labels=num_labels, transform_func=transform_func)
    classifier_train_data, classifier_train_labels = extract_features(vae_model, classifier_train_loader, device)
    test_data, test_labels = extract_features(vae_model, test_loader, device)

    # Model
    svm_classifier = SVC(kernel=kernel)
    svm_classifier.fit(classifier_train_data, classifier_train_labels)

    predictions = svm_classifier.predict(test_data)
    svm_acc = np.mean(predictions == test_labels)

    print(f"SVM: Accuracy: {(svm_acc * 100):.2f}%, kernel={kernel}, labels={num_labels}")

    return svm_classifier, svm_acc


def optimize_lr(labels, params):
    params['latent_epochs'] = 5

    lrs = [random.uniform(1.0e-4, 1.0e-3) for _ in range(30)]

    min_loss = np.inf
    best_lr = params['lr']
    for _lr in lrs:
        print(f"Optimizing LR: Testing LR={_lr}")

        params['lr'] = _lr

        try:
            _, loss = fit_vae(num_labels=labels,
                              is_save_tensorboard=False,
                              params=params)
        except:
            loss = np.inf

        if loss < min_loss:
            min_loss = loss
            best_lr = _lr

    print(f"Optimizing LR: labels={labels}, best LR={best_lr}")

    return best_lr


# Q4

# Function to display images
def show_images(images, title):
    plt.figure(figsize=(10, 1))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(vutils.make_grid(images, padding=2, normalize=True), (1, 2, 0)))
    plt.show()


# Define a function to generate images from trained models
def generate_images(generator, num_images, nz):
    noise = torch.randn(num_images, nz, 1, 1, device=device)
    generated_images = generator(noise).detach().cpu()
    return generated_images


# Define a function to plot loss functions
def plot_losses(d_losses, g_losses, title):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# Define the training loop
def train_dcgan(nz, dataloader, netG, netD, criterion, optimizerG, optimizerD, num_epochs=5):
    # Define the device
    d_losses = []
    g_losses = []
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()
            real_images = data[0].to(device)
            batch_size = real_images.size(0)
            label = torch.full((batch_size,), 1, dtype=torch.float, device=device)  # Real label
            output = netD(real_images).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(0)  # Fake label
            output = netD(fake_images.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update Generator network: maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(1)  # Real label
            output = netD(fake_images).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            d_losses.append(errD.item())
            g_losses.append(errG.item())

        return d_losses, g_losses


# Define the training loop
def train_wgan(nz, dataloader, netG, netD, optimizerG, optimizerD, num_epochs=5):
    d_losses = []
    g_losses = []
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # Train Discriminator
            netD.zero_grad()
            real_images = data[0].to(device)
            batch_size = real_images.size(0)
            real_output = netD(real_images).view(-1)
            fake_images = netG(torch.randn(batch_size, nz, 1, 1, device=device))
            fake_output = netD(fake_images.detach()).view(-1)
            gradient_penalty = calculate_gradient_penalty(netD, real_images, fake_images)
            d_loss = wasserstein_loss(fake_output, -torch.ones_like(fake_output)) - \
                     wasserstein_loss(real_output, torch.ones_like(real_output)) + \
                     gradient_penalty
            d_loss.backward()
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            fake_output = netD(fake_images).view(-1)
            g_loss = -wasserstein_loss(fake_output, torch.ones_like(fake_output))
            g_loss.backward()
            optimizerG.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        return d_losses, g_losses


def dcgan():
    # Initialize hyperparameters
    nz = 100  # latent z vector
    ngf = 64  # feature maps in generator
    ndf = 64  # feature maps in discriminator
    nc = 1  # number of channels in the output images (for MNIST dataset)

    # Create the generator
    netG = DCGAN_Generator(nz, ngf, nc)
    # Create the discriminator
    netD = DCGAN_Discriminator(nc, ndf)

    # Define loss function (Binary Cross Entropy)
    criterion = nn.BCELoss()

    # Setup Adam optimizers for both G and D
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Apply the weights_init function to both networks
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Load the MNIST dataset
    train_loader, test_loader = load_Fashionmnist(batch_size=128, architecture='DCGAN')

    # Train the DCGAN
    netG.train()
    netD.train()
    dcgan_d_losses, dcgan_g_losses = train_dcgan(nz, train_loader, netG, netD, criterion, optimizerG, optimizerD, num_epochs=5)
    plot_losses(dcgan_d_losses, dcgan_g_losses, 'DCGAN Losses (MNIST)')

    # Generate images using DCGAN
    netG.eval()
    num_images_to_generate = 10
    dcgan_generated_images = generate_images(netG, num_images_to_generate, nz)

    # Show DCGAN generated images
    show_images(dcgan_generated_images, "DCGAN Generated Images")


def wgan():
    # Initialize hyperparameters
    nz = 100  # latent z vector
    ngf = 64  # feature maps in generator
    ndf = 64  # feature maps in discriminator
    nc = 1  # number of channels in the output images (for MNIST dataset)

    # Create the generator
    netG = WGAN_Generator(nz, ngf, nc)

    # Create the discriminator
    netD = WGAN_Discriminator(nc, ndf)

    # loss is: def wasserstein_loss(output, target)

    # Setup Adam optimizers for both G and D
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Load the MNIST dataset
    train_loader, test_loader = load_Fashionmnist(batch_size=128, architecture='WGAN')

    # Train the WGAN
    netG.train()
    netD.train()
    wgan_d_losses, wgan_g_losses = train_wgan(nz, train_loader, netG, netD, optimizerG, optimizerD, num_epochs=5)
    plot_losses(wgan_d_losses, wgan_g_losses, 'WGAN Losses (Fashion MNIST)')

    # Generate images using WGAN
    netG.eval()
    num_images_to_generate = 10
    wgan_generated_images = generate_images(netG, num_images_to_generate, nz)

    # Show WGAN generated images
    show_images(wgan_generated_images, "WGAN Generated Images")

def cifar_10():
    for _ in CIFAR10_ARCHITECTURE:
        if _ == 'DCGAN':
            dcgan()

        elif _ == 'WGAN':
            wgan()



def main():
    seed_handler._set_seed(INIT_SEED)

    vae_models = dict()
    svm_models = dict()

    params = {'lr': 2.0e-3,
              'latent_epochs': 30,
              'classifier_epochs': 10,
              'batch_size': 64,
              'optimizer': 'adam',
              'lr_scheduler': {
                  'enable': True,
                  'gamma': 0.5,
                  'step': 5
              }}

    if IS_OPTIMIZE_LR:
        best_lrs = {}
        for _labels in LABELS:
            best_lrs[_labels] = optimize_lr(_labels, params.copy())
            print(best_lrs)

        return

    for _labels in LABELS:
        vae_models[_labels], loss = fit_vae(num_labels=_labels,
                                            is_save_tensorboard=True,
                                            params=params)

        model_utils.save_model(model=vae_models[_labels],
                               output_path=os.path.join(MODELS_OUTPUT_DIR,
                                                        f"vae_{_labels}_labels_datatype_{DATA_TYPE}_model.pth"))

        vae_models[_labels] = model_utils.load_model(model_path=os.path.join(MODELS_OUTPUT_DIR,
                                                                             f"vae_{_labels}_labels_datatype_{DATA_TYPE}_model.pth"))

        svm_models[_labels], loss = fit_svm(num_labels=_labels,
                                            kernel='rbf',
                                            vae_model=vae_models[_labels])

        model_utils.save_model(model=svm_models[_labels],
                               output_path=os.path.join(MODELS_OUTPUT_DIR,
                                                        f"svm_{_labels}_labels_datatype_{DATA_TYPE}_model.pth"))

        print("\n")

    cifar_10()

    pass


if __name__ == '__main__':
    main()
