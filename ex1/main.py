import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

from configs import DATA_DIR, MODEL_TYPES, INIT_SEED, OUTPUTS_DIR
from utils import mnist_reader, seed_handler
from model import Lenet5
from data_handler import Lenet5_Dataset

def _load_data(batch_size):
    X_train, y_train = mnist_reader.load_mnist(DATA_DIR, kind='train')
    X_test, y_test = mnist_reader.load_mnist(DATA_DIR, kind='t10k')

    train_loader = DataLoader(dataset=Lenet5_Dataset(X_train, y_train),
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=Lenet5_Dataset(X_test, y_test),
                              batch_size=batch_size,
                              shuffle=False)

    return train_loader, test_loader

def fit(model_type, optimizer_method, params):
    weight_decay = params['weight_decay']
    lr = params['lr']
    epochs = params['epochs']
    batch_size = params['batch_size']

    # Set results directory
    results_dir = os.path.join(OUTPUTS_DIR, f"model_{model_type}")

    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set Data
    train_loader, test_loader = _load_data(batch_size)
    total_steps = len(train_loader)

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

    # TensorBoard
    writer = SummaryWriter(log_dir=results_dir)

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

            # Forward
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Calculate Accuracy
            epoch_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            epoch_correct += (predictions == labels).sum().item()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss /= len(train_loader.dataset)
        train_loss_history.append(epoch_loss)
        epoch_acc = epoch_correct / len(train_loader.dataset)
        train_acc_history.append(epoch_acc)

        # Log metrics to TensorBoard
        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_acc, epoch)

        print(f"Model: {model_type}, Train: epoch {epoch + 1}/{epochs}, Loss = {epoch_loss:.6f}, Accuracy {epoch_acc}")

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

            print(f"Model: {model_type}, Validation: epoch {epoch + 1}/{epochs}, Loss = {epoch_loss:.6f}, Accuracy {epoch_acc}")

    writer.close()
    pass

def main():
    seed_handler._set_seed(INIT_SEED)

    models = dict()
    params = {'weight_decay': 0.0,
              'lr': 1e-5,
              'epochs': 50,
              'batch_size': 32}

    for _model_type in MODEL_TYPES:
        if _model_type == 'weight_decay':
            params['weight_decay'] = 1.0e-2
        else:
            params['weight_decay'] = 0.0

        models[_model_type] = fit(model_type=_model_type,
                                  optimizer_method='adam',
                                  params=params)

        print("\n")

    pass


if __name__ == '__main__':
    main()
