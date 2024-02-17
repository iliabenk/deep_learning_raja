import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

class EnsembleModel(nn.Module):
    def __init__(self, models_list):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models_list)

    def forward(self, x, input_states):
        outputs, states = [], []

        for _input_states, _model in zip(input_states, self.models):
            cur_outputs, cur_states = _model(x, _input_states)
            outputs.append(cur_outputs)
            states.append(cur_states)

        combined_outputs = torch.mean(torch.stack(outputs), dim=0)

        return combined_outputs, states

class Model(nn.Module):
    def __init__(self, hidden_size, output_size, dropout, num_layers, embed_dim, model_type,
                 pretrained_embeddings=None):
        super(Model, self).__init__()
        self.model_type = model_type

        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=embed_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.stack(pretrained_embeddings), freeze=False)

        self.dropout = nn.Dropout(p=dropout)

        if model_type == "LSTM":
            self.model = nn.LSTM(input_size=embed_dim,
                                hidden_size=hidden_size,
                                batch_first=True,
                                num_layers=num_layers,
                                dropout=dropout)

            # self.model = [nn.LSTM(input_size=hidden_size,
            #                     hidden_size=hidden_size,
            #                     batch_first=True,
            #                     dropout=dropout) for _ in range(num_layers)]
        elif model_type == "GRU":
            self.model = nn.GRU(input_size=embed_dim,
                                hidden_size=hidden_size,
                                batch_first=True,
                                num_layers=num_layers,
                                dropout=dropout)
        else:
            assert False, f"Unsupported model_type={model_type}"

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, states):
        if isinstance(states, (tuple, list)):
            states = [e.detach() for e in states]
        else:
            states = states.detach()

        out = self.embedding(x)

        out, states_o = self.model(out, states)

        out = self.fc(self.dropout(out))

        return out, states_o

    # def forward(self, x, states):
    #     out = self.embedding(x)
    #
    #     for _layer in self.model:
    #         out, states = _layer(out, states)
    #
    #     if isinstance(states, (tuple, list)):
    #         states = [e.detach() for e in states]
    #     else:
    #         states = states.detach()
    #
    #     out = self.fc(out)
    #
    #     return out, states

    def init_states(self, batch_size, device):
        if self.model_type == "LSTM":
            h = torch.zeros(self.model.num_layers, batch_size, self.model.hidden_size, device=device, requires_grad=False)
            c = torch.zeros(self.model.num_layers, batch_size, self.model.hidden_size, device=device, requires_grad=False)

            # h = torch.zeros(1, batch_size, self.model[0].hidden_size, device=device, requires_grad=False)
            # c = torch.zeros(1, batch_size, self.model[0].hidden_size, device=device, requires_grad=False)

            states = (h, c)

        elif self.model_type == "GRU":
            h = torch.zeros(self.model.num_layers, batch_size, self.model.hidden_size, device=device, requires_grad=False)
            states = (h)

        return states



