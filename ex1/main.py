import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

from configs import MODEL_TYPES, INIT_SEED, TENSORBOARD_DIR, MODELS_OUTPUT_DIR
from utils import seed_handler, model_utils
from model import Lenet5
from data_handler import load_data, transform_func

def fit(model_type, optimizer_method, params):
    weight_decay = params['weight_decay']
    lr = params['lr']
    epochs = params['epochs']
    batch_size = params['batch_size']
    lr_scheduler_enabled = params['lr_scheduler']['enable']
    lr_scheduler_gamma = params['lr_scheduler']['gamma']
    lr_scheduler_step = params['lr_scheduler']['step']

    # Set tensorboard directory
    if lr_scheduler_enabled:
        tensorboard_dir = os.path.join(TENSORBOARD_DIR, f"model_{model_type}__epochs_{epochs}__lr_{lr}__weightDecay_"
                                                        f"{weight_decay}__batchSize_{batch_size}__optimizer_"
                                                        f"{optimizer_method}__lr_scheduler_params__enabled_"
                                                        f"{lr_scheduler_enabled}__gamma_{lr_scheduler_gamma}__step_"
                                                        f"{lr_scheduler_step}")

    else:
        tensorboard_dir = os.path.join(TENSORBOARD_DIR, f"model_{model_type}__epochs_{epochs}__lr_{lr}__weightDecay_"
                                                        f"{weight_decay}__batchSize_{batch_size}__optimizer_"
                                                        f"{optimizer_method}__lr_scheduler_enabled_{lr_scheduler_enabled}")

    # TensorBoard
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set Data
    train_loader, test_loader = load_data(batch_size, transform_func=transform_func)

    # Model
    model = Lenet5(model_type=model_type,
                   num_classes=10).to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if optimizer_method == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_method == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        assert False, f"optimizer={optimizer_method} is not supported"

    print(f"Optimizer = {optimizer_method}")

    scheduler = None
    if lr_scheduler_enabled:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step,
                                                    gamma=lr_scheduler_gamma)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    model.train()
    for epoch in range(epochs):
        epoch_correct = 0
        epoch_loss = 0
        for i, (data, labels) in enumerate(train_loader):
            data = data.float().to(device)
            labels = labels.to(device)

            # For dropout do Forward twice.
            # Once for loss & accuracy without dropout, and again with dropout for training
            if model_type == "dropout":
                model.eval()
                with torch.no_grad():
                    # Forward
                    outputs = model(data)
                    loss = criterion(outputs, labels)

                    # Calculate Accuracy
                    epoch_loss += loss.item()
                    _, predictions = torch.max(outputs, 1)
                    epoch_correct += (predictions == labels).sum().item()

                model.train()

            # Forward
            outputs = model(data)
            loss = criterion(outputs, labels)

            if model_type != "dropout":
                # Calculate Accuracy
                epoch_loss += loss.item()
                _, predictions = torch.max(outputs, 1)
                epoch_correct += (predictions == labels).sum().item()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        epoch_loss /= len(train_loader.dataset)
        train_loss_history.append(epoch_loss)
        epoch_acc = epoch_correct / len(train_loader.dataset)
        train_acc_history.append(epoch_acc)

        # Log metrics to TensorBoard
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_acc, epoch)

        print(f"Model: {model_type}, Train: epoch [{epoch + 1}/{epochs}], Loss = {epoch_loss:.6f}, Accuracy"
              f" {epoch_acc}")

        # Validation
        model.eval()
        with torch.no_grad():
            epoch_correct = 0
            epoch_loss = 0

            for i, (data, labels) in enumerate(test_loader):
                data = data.float().to(device)
                labels = labels.to(device)

                # Forward
                outputs = model(data)
                loss = criterion(outputs, labels)

                # Calculate Accuracy
                epoch_loss += loss.item()
                _, predictions = torch.max(outputs, 1)
                epoch_correct += (predictions == labels).sum().item()

            epoch_loss /= len(test_loader.dataset)
            val_loss_history.append(epoch_loss)
            epoch_acc = epoch_correct / len(test_loader.dataset)
            val_acc_history.append(epoch_acc)

            writer.add_scalar('Validation/Loss', epoch_loss, epoch)
            writer.add_scalar('Validation/Accuracy', epoch_acc, epoch)
            writer.add_scalar('GE/Accuracy (train - test)', train_acc_history[-1] - val_acc_history[-1], epoch)
            writer.add_scalar('GE/Loss (test - train)', val_loss_history[-1] - train_loss_history[-1], epoch)

            print(f"Model: {model_type}, Validation: epoch [{epoch + 1}/{epochs}], Loss = {epoch_loss:.6f}, "
                  f"Accuracy {epoch_acc}")

    writer.close()

    return model

def main():
    seed_handler._set_seed(INIT_SEED)

    models = dict()
    params = {'weight_decay': 0.0,
              'lr': 1.0e-3,
              'epochs': 10,
              'batch_size': 64,
              'lr_scheduler': {
                  'enable': True,
                  'gamma': 0.5,
                  'step': 3
              }}

    for _model_type in MODEL_TYPES:
        if _model_type == 'weight_decay':
            params['weight_decay'] = 1.0e-3
        else:
            params['weight_decay'] = 0.0

        models[_model_type] = fit(model_type=_model_type,
                                  optimizer_method='adam',
                                  params=params)

        model_utils.save_model(model=models[_model_type],
                               output_path=os.path.join(MODELS_OUTPUT_DIR, f"{_model_type}_model.pth"))

        print("\n")

    pass


if __name__ == '__main__':
    main()
