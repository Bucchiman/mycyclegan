import random
import time
import datetime
import sys
import json
from pathlib import Path
import torch

from torchvision.utils import save_image


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


class Config(object):
    def __init__(self):
        pass

    def save_config(self, args, output_path):
        with open(str(Path(output_path).joinpath("config.json")), 'w') as fp:
            json.dump(args, fp, indent=4)

    def load_config(self, config_path):
        with open(str(Path(config_path)), 'r') as fp:
            args = json.load(fp)
        return args


def save_model(model, save_models_path: str):
    torch.save(model.state_dict(), save_models_path)
