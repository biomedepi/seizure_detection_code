from dataclasses import dataclass, field
import os
import pickle
import tensorflow as tf

@dataclass
class Settings:
    name: str
    seed: int = 12345

    # Model parameters
    frame: float = 2
    stride: float = 1
    stride_s: float = 0.5
    batch_size: int = 16
    factor: int = 5
    act: str = None
    lr: float = 0.0001
    l2: float = 0
    dropoutRate: float = 0
    nr_epochs: int = 50
    class_weights: dict = field(default_factory=dict)
    

    def save(self, path, filename: str):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, filename), 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


    def load(filename: str):
        if not os.path.exists(filename):
            raise ValueError('File does not exist')

        with open(filename, 'rb') as input:
            settings = pickle.load(input)

        return settings
