import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
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


############### Q4 ###############

#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

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
    # Define DCGAN parameters
    # Number of workers for dataloader
    workers = 1
    # Batch size during training
    batch_size = 128
    # Spatial size of training images. All images will be resized to this
    image_size = 64
    # Number of color channels in the training images.
    nc = 1 #Gray
    # Size of z latent vector (i.e. size of generator input)
    nz = 100
    # Size of feature maps in generator
    ngf = 32
    # Size of feature maps in discriminator
    ndf = 32
    # Number of training epochs
    num_epochs = 5
    # Learning rate for optimizers
    lr = 0.0002
    # Beta1 hyperparameter for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = torch.cuda.device_count()
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Load the MNIST dataset
    train_loader, test_loader = load_Fashionmnist(batch_size=128, architecture='DCGAN')

    # Create the generator
    netG = DCGAN_Generator(ngpu, nz, ngf, nc).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    print(netG)

    # Create the Discriminator
    netD = DCGAN_Discriminator(ngpu, nc, ndf).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    print(netD)

    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(image_size, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(train_loader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(train_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(train_loader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    # Results
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())
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

def DCGAN_WGAN():
    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results

    for _ in CIFAR10_ARCHITECTURE:
        if _ == 'DCGAN':
            dcgan()
            assert False

        elif _ == 'WGAN':
            wgan()



def main():
    # seed_handler._set_seed(INIT_SEED)
    #
    # vae_models = dict()
    # svm_models = dict()
    #
    # params = {'lr': 2.0e-3,
    #           'latent_epochs': 30,
    #           'classifier_epochs': 10,
    #           'batch_size': 64,
    #           'optimizer': 'adam',
    #           'lr_scheduler': {
    #               'enable': True,
    #               'gamma': 0.5,
    #               'step': 5
    #           }}
    #
    # if IS_OPTIMIZE_LR:
    #     best_lrs = {}
    #     for _labels in LABELS:
    #         best_lrs[_labels] = optimize_lr(_labels, params.copy())
    #         print(best_lrs)
    #
    #     return
    #
    # for _labels in LABELS:
    #     vae_models[_labels], loss = fit_vae(num_labels=_labels,
    #                                         is_save_tensorboard=True,
    #                                         params=params)
    #
    #     model_utils.save_model(model=vae_models[_labels],
    #                            output_path=os.path.join(MODELS_OUTPUT_DIR,
    #                                                     f"vae_{_labels}_labels_datatype_{DATA_TYPE}_model.pth"))
    #
    #     vae_models[_labels] = model_utils.load_model(model_path=os.path.join(MODELS_OUTPUT_DIR,
    #                                                                          f"vae_{_labels}_labels_datatype_{DATA_TYPE}_model.pth"))
    #
    #     svm_models[_labels], loss = fit_svm(num_labels=_labels,
    #                                         kernel='rbf',
    #                                         vae_model=vae_models[_labels])
    #
    #     model_utils.save_model(model=svm_models[_labels],
    #                            output_path=os.path.join(MODELS_OUTPUT_DIR,
    #                                                     f"svm_{_labels}_labels_datatype_{DATA_TYPE}_model.pth"))
    #
    #     print("\n")

    DCGAN_WGAN()

    pass


if __name__ == '__main__':
    main()
