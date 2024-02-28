import os.path

import torch

def save_model(model, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    torch.save(model, output_path)

def save_model_state_dict(model, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    torch.save(model.state_dict(), output_path)

def load_model_state_dict(model, model_path, device):
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)

    return model

def load_model(model_path, device='cpu'):
    model = torch.load(model_path, map_location=torch.device(device))

    return model
