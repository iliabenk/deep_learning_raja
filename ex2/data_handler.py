import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

from utils import mnist_reader
from configs import TRAIN_DATA, VAL_DATA, TEST_DATA

def load_data(batch_size=20, seq_len=35):
    num_first_whitespaces = 1

    with open(TRAIN_DATA) as f:
        file = f.read()
        train_data = file[num_first_whitespaces:].split(' ')
    with open(VAL_DATA) as f:
        file = f.read()
        val_data = file[num_first_whitespaces:].split(' ')
    with open(TEST_DATA) as f:
        file = f.read()
        test_data = file[num_first_whitespaces:].split(' ')

    vocabulary = sorted(set(train_data))

    char2ind = {c: i for i, c in enumerate(vocabulary)}

    train_indices = np.array([char2ind[c] for c in train_data]).reshape(-1, 1)
    val_indices = np.array([char2ind[c] for c in val_data]).reshape(-1, 1)
    test_indices = np.array([char2ind[c] for c in test_data]).reshape(-1, 1)

    return (batch(data=train_indices, batch_size=batch_size, seq_len=seq_len),
            batch(data=val_indices, batch_size=batch_size, seq_len=seq_len),
            batch(data=test_indices, batch_size=batch_size, seq_len=seq_len),
            vocabulary)


def batch(data, batch_size=20, seq_len=35):
    data = torch.tensor(data, dtype=torch.int64)

    num_batches = int(data.size(0) / batch_size)

    data = data[:num_batches * batch_size]
    data = data.view(batch_size, -1)

    dataset = []
    for i in range(0, data.size(1) - 1, seq_len):
        seq_len_adj = min(seq_len, data.size(1) - 1 - i)

        x = data[:, i:i + seq_len_adj]
        y = data[:, i + 1:i + seq_len_adj + 1]
        dataset.append((x, y))

    return dataset

def load_glove_weights(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]])
            embeddings[word] = vector
    return embeddings


# # Specify the path to the GloVe file
# glove_file = 'path/to/glove.6B.300d.txt'
#
# # Load GloVe embeddings
# glove_embeddings = load_glove_embeddings(glove_file)
#
# # Extract vocabulary size and embedding dimension
# vocab_size, embedding_dim = len(glove_embeddings), len(next(iter(glove_embeddings.values())))
#
# # Initialize the embedding layer with GloVe embeddings
# embedding_layer = nn.Embedding.from_pretrained(torch.stack(list(glove_embeddings.values())), freeze=False)

if __name__ == "__main__":
    load_data()
