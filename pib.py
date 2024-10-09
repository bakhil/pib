import torch
import torch.nn as nn
import lightning as L
from dataset import PIBDataset
from model import get_model
from lightning.pytorch.cli import LightningArgumentParser
import utils

if __name__ == '__main__':
    parser = LightningArgumentParser(default_config_files=['config.yaml'])
    parser.add_argument()
    args = parser.parse_args()