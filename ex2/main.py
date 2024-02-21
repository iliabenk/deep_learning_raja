import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from itertools import product
import timeit

from configs import MODEL_TYPES, INIT_SEED, TENSORBOARD_DIR, MODELS_OUTPUT_DIR, DROPOUT_VALS, WORD_PRETRAINED_EMBED_FILE
from utils import seed_handler, model_utils
from model import Model, EnsembleModel
from data_handler import load_data, load_glove_weights

def  fit(model_type, optimizer_method, use_pretrained_embeddings, ensemble_size, factor_epoch, lr_factor,
        dropout, lr, epochs, batch_size, seq_len, lr_schedule_enabled, winit):

    # Set tensorboard directory
    tensorboard_dir = os.path.join(TENSORBOARD_DIR, f"model_{model_type}__epochs_{epochs}__lr_{lr}__optimizer_"
                                                        f"{optimizer_method}__dropout_{dropout}")

    # TensorBoard
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set Data
    train_dataset, val_dataset, test_dataset, vocab = load_data(batch_size=batch_size, seq_len=seq_len)

    # Model
    print(f"Model = {model_type}, dropout = {dropout}")

    if use_pretrained_embeddings:
        pretrained_embeddings = load_glove_weights(WORD_PRETRAINED_EMBED_FILE)

        pretrained_embeddings_ordered = []
        for _word in vocab:
            if _word not in pretrained_embeddings.keys():
                print(f"Missing word embedding: {_word}. Setting random weights")
                pretrained_embeddings_ordered.append(torch.randn(size=list(pretrained_embeddings.values())[
                    0].shape))
            else:
                pretrained_embeddings_ordered.append(pretrained_embeddings[_word])

    else:
        pretrained_embeddings_ordered = None

    models_list = []
    for _ in range(ensemble_size):
        cur_model = Model(hidden_size=200,
                          output_size=len(vocab),
                          dropout=dropout,
                          embed_dim=200,
                          num_layers=2,
                          model_type=model_type,
                          pretrained_embeddings=pretrained_embeddings_ordered).to(device)

        models_list.append(cur_model)

    model = EnsembleModel(models_list)

    # Init weights to [-0.1, 0.1] uniformly as in the paper
    if winit is not None:
        for param in model.parameters():
            if len(param.size()) >= 2:
                # If the parameter is a weight matrix (not bias), apply uniform initialization
                nn.init.uniform_(param, a=-winit, b=winit)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if optimizer_method == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_method == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        assert False, f"optimizer={optimizer_method} is not supported"

    print(f"Optimizer = {optimizer_method}")

    tic = timeit.default_timer()
    for epoch in range(epochs):
        model.train()

        if lr_schedule_enabled and epoch >= factor_epoch:
            lr /= lr_factor

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        states = [model.models[0].init_states(batch_size=batch_size, device=device) for _ in range(ensemble_size)]
        epoch_loss = 0
        for i, (data, labels) in enumerate(train_dataset):
            data = data.to(device)
            labels = labels.to(device)

            # Forward
            outputs, states = model(data, states)

            # outputs is of shape (batch_size, seq_len, len(vocab)), but should be (batch_size, len(vocab), seq_len)
            # Multiply by batch_size because this is what they did in the paper (alternatively can modify LR)
            loss = criterion(outputs.transpose(1, 2), labels) * batch_size

            # Calculate Loss
            epoch_loss += loss.item()

            # Backward
            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5 * ensemble_size)

            optimizer.step()

            if i % (len(train_dataset)//10) == 0:
                toc = timeit.default_timer()
                print(f"Training: epoch={epoch}/{epochs}, step={i}/{len(train_dataset)}, lr={lr}, "
                      f"train_loss={loss.item() / batch_size}, "
                      f"norm={norm},"
                      f"Training time so far: {round((toc-tic) / 60)} mins")

        # Train Perplexity
        train_perplexity = perplexity(model, train_dataset, device, batch_size, ensemble_size, criterion)

        writer.add_scalar('Train/Perplexity', train_perplexity, epoch)

        print(f"Model: {model_type}, dropout: {dropout}, Train: epoch [{epoch + 1}/{epochs}], "
              f"Perplexity {train_perplexity}")

        # Validation Perplexity
        val_perplexity = perplexity(model, val_dataset, device, batch_size, ensemble_size, criterion)

        writer.add_scalar('Validation/Perplexity', val_perplexity, epoch)

        print(f"Model: {model_type}, dropout: {dropout}, Validation: epoch [{epoch + 1}/{epochs}], "
              f"Perplexity {val_perplexity}")

        # Test Perplexity
        test_perplexity = perplexity(model, test_dataset, device, batch_size, ensemble_size, criterion)

        writer.add_scalar('Test/Perplexity', test_perplexity, epoch)

        print(f"Model: {model_type}, dropout: {dropout}, Test: epoch [{epoch + 1}/{epochs}], "
              f"Perplexity {test_perplexity}")

    writer.close()

    return model

def perplexity(model, dataset, device, batch_size, ensemble_size, criterion):
    with torch.no_grad():
        model.eval()
        states = [model.models[0].init_states(batch_size=batch_size, device=device) for _ in range(ensemble_size)]
        total_loss = 0

        for i, (data, labels) in enumerate(dataset):
            data = data.to(device)
            labels = labels.to(device)

            # Forward
            outputs, states = model(data, states)

            # outputs is of shape (batch_size, seq_len, len(vocab)), but should be (batch_size, len(vocab), seq_len)
            loss = criterion(outputs.transpose(1, 2), labels)

            # Calculate Loss
            total_loss += loss.item()

        total_loss /= len(dataset)

        return torch.exp(torch.tensor(total_loss))

def main():
    seed_handler._set_seed(INIT_SEED)

    models = dict()
    models["GRU_dropout"] = fit(model_type='GRU',
                              optimizer_method='sgd',
                              use_pretrained_embeddings=True,
                              ensemble_size=1,
                              factor_epoch=40,
                              lr_factor=1.1,
                              dropout=0.5,
                              lr=0.2,
                              epochs=70,
                              batch_size=20,
                              seq_len=35,
                              lr_schedule_enabled=True,
                              winit=None)

    models["LSTM_dropout"] = fit(model_type='LSTM',
                              optimizer_method='sgd',
                              use_pretrained_embeddings=False,
                              ensemble_size=1,
                              factor_epoch=40,
                              lr_factor=2,
                              dropout=0.5,
                              lr=1,
                              epochs=50,
                              batch_size=20,
                              seq_len=35,
                              lr_schedule_enabled=True,
                              winit=0.05)

    models["LSTM_no_dropout"] = fit(model_type='LSTM',
                              optimizer_method='sgd',
                              use_pretrained_embeddings=False,
                              ensemble_size=1,
                              factor_epoch=4,
                              lr_factor=2,
                              dropout=0.0,
                              lr=1.0e-0,
                              epochs=13,
                              batch_size=20,
                              seq_len=35,
                              lr_schedule_enabled=True,
                              winit=0.05)

    models["GRU_no_dropout"] = fit(model_type='GRU',
                              optimizer_method='sgd',
                              use_pretrained_embeddings=False,
                              ensemble_size=1,
                              factor_epoch=7,
                              lr_factor=1.5,
                              dropout=0.0,
                              lr=2.0e-1,
                              epochs=13,
                              batch_size=20,
                              seq_len=35,
                              lr_schedule_enabled=True,
                              winit=0.1)

    for _model_type, _model in models.items():
        model_utils.save_model(model=_model,
                               output_path=os.path.join(MODELS_OUTPUT_DIR, f"{_model_type}_model.pth"))

    print("\n")

    pass


if __name__ == '__main__':
    main()
