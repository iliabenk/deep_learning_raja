import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import os
import timeit
from torch.nn import functional as F
import random
from sklearn.svm import SVC
import torchvision.utils as vutils
import torch.optim as optim

from configs import LABELS, INIT_SEED, TENSORBOARD_DIR, MODELS_OUTPUT_DIR, IS_OPTIMIZE_LR, DATA_TYPE
from dcgan import DCGAN_MODEL
from wgan import WGAN_GP
from utils import seed_handler, model_utils
from model import VAE
from data_handler import load_data, transform_func, extract_features, get_data_loader, gan_type

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

def WGAN_CP(args):
    pass


def gan(gan_type):
    model = None
    if gan_type.model == 'DCGAN':
        model = DCGAN_MODEL(gan_type)
    elif gan_type.model == 'WGAN_GP':
        model = WGAN_GP(gan_type)
    else:
        print("Model type non-existing. Try again.")
        exit(-1)

    # Load datasets to train and test loaders
    train_loader, test_loader = get_data_loader(gan_type)
    # feature_extraction = FeatureExtractionTest(train_loader, test_loader, args.cuda, args.batch_size)

    # Start model training
    if gan_type.is_train == 'True':
        print('training')
        model.train(train_loader)

    # start evaluating on test data
    else:
        print('evaluating')
        model.evaluate(test_loader, gan_type.load_D, gan_type.load_G)
        for i in range(50):
            model.generate_latent_walk(i)

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

    if not torch.cuda.is_available():
        print('no cuda')
    dcgan = gan_type(model='DCGAN', is_train='True', download='True', dataroot='datasets/fashion-mnist',
                     dataset='fashion-mnist', epochs=5, batch_size=64)

    wgan = gan_type(model='WGAN_GP', is_train='True', download='True', dataroot='datasets/fashion-mnist',
                    dataset='fashion-mnist', epochs=5, batch_size=64)

    print(f'DCGAN configuration is:\n {dcgan}\n\n\n')
    print(f'WGAN configuration is:\n {wgan}\n\n\n')

    # Train
    # gan(dcgan)
    # gan(wgan)

    # Evaluate
    dcgan.is_train = wgan.is_train = 'False'
    gan(dcgan)
    gan(wgan)

    pass


if __name__ == '__main__':
    main()
