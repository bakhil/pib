import torch
import torch.nn as nn
import lightning as L
from dataset import PIBDataset
from model import get_model
from torch.utils.data import DataLoader
import utils

if __name__ == '__main__':
    parser = utils.get_parser()
    # Required arguments on command line are --config and --mode
    args = parser.parse_args()

    if args.lightning_seed is not None:
        L.seed_everything(args.lightning_seed)

    pib_model = get_model(args.model_name, **args.model)

    dataset = PIBDataset(mode=args.mode, **args.data)
    loader = DataLoader(dataset)
    
    if args.mode == 'train':
        dataset_val = PIBDataset(mode='validate', **args.data)
        loader_val = DataLoader(dataset_val)
        trainer = L.Trainer(max_epochs=args.train.max_epochs, default_root_dir=args.train.root_dir)
        if args.model.fresh_init:
            trainer.fit(pib_model, loader, loader_val)
        elif args.model.from_checkpoint is not None:
            trainer.fit(pib_model, loader, loader_val, ckpt_path=args.model.from_checkpoint)
        else:
            trainer.fit(pib_model, loader, loader_val, ckpt_path='last')