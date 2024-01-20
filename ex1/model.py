import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class Lenet5(nn.Module):
    def __init__(self, model_type: str, num_classes: int):
        super(Lenet5, self).__init__()

        self.layers_list = self._init_model(model_type=model_type, num_classes=num_classes)

    @staticmethod
    def _init_model(model_type, num_classes):
        if model_type == 'regular':
            return Lenet5._init_model_regular(num_classes)
        elif model_type == 'dropout':
            return Lenet5._init_model_dropout(num_classes)
        elif model_type == 'weight_decay':
            return Lenet5._init_model_weight_decay(num_classes)
        elif model_type == 'bn':
            return Lenet5._init_model_bn(num_classes)
        else:
            assert False, f"Unsupported model_type={model_type}"

    @staticmethod
    def _init_model_regular(num_classes):
        layers_list = nn.ModuleList()

        layers_list.append(nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                                         nn.MaxPool2d(kernel_size=2, stride=2)))

        layers_list.append(nn.Sequential(nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                                         nn.MaxPool2d(kernel_size=2, stride=2)))

        layers_list.append(nn.Flatten())

        layers_list.append(nn.Sequential(nn.Linear(256, 84),
                                         nn.ReLU()))

        layers_list.append(nn.Linear(84, num_classes))

        return layers_list

    @staticmethod
    def _init_model_dropout(num_classes):
        layers_list = nn.ModuleList()

        layers_list.append(nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                                         nn.MaxPool2d(kernel_size=2, stride=2)))

        layers_list.append(nn.Sequential(nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                                         nn.MaxPool2d(kernel_size=2, stride=2)))

        layers_list.append(nn.Flatten())

        layers_list.append(nn.Sequential(nn.Linear(256, 84),
                                         nn.ReLU(),
                                         nn.Dropout(0.5)))

        layers_list.append(nn.Linear(84, num_classes))

        return layers_list

    @staticmethod
    def _init_model_weight_decay(num_classes):
        return Lenet5._init_model_regular(num_classes=num_classes)

    @staticmethod
    def _init_model_bn(num_classes):
        layers_list = nn.ModuleList()

        layers_list.append(nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                                         nn.BatchNorm2d(6),
                                         nn.MaxPool2d(kernel_size=2, stride=2)))

        layers_list.append(nn.Sequential(nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                                         nn.BatchNorm2d(16),
                                         nn.MaxPool2d(kernel_size=2, stride=2)))

        layers_list.append(nn.Flatten())

        layers_list.append(nn.Sequential(nn.Linear(256, 84),
                                         nn.ReLU()))

        layers_list.append(nn.Linear(84, num_classes))

        return layers_list

    def forward(self, x):
        out = x

        for layer in self.layers_list:
            out = layer(out)

        return out
