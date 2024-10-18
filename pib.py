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
    
    if args.mode == 'train':
        pib_model = get_model(args.model.name, **args.model)
        
        dataset = PIBDataset(mode=args.mode, data_path=args.data.data_path, train_validate_test_split=args.data.train_validate_test_split)
        loader = DataLoader(dataset, batch_size=args.data.train_batch_size, shuffle=True)
        dataset_val = PIBDataset(mode='validate', data_path=args.data.data_path, train_validate_test_split=args.data.train_validate_test_split)
        loader_val = DataLoader(dataset_val, batch_size=args.data.validate_batch_size, shuffle=False)

        trainer = L.Trainer(max_epochs=args.train.max_epochs, default_root_dir=args.train.root_dir, 
                                profiler='simple', accumulate_grad_batches=args.train.accumulate_grad_batches,
                                precision=args.train.precision)
        if args.model.from_checkpoint is not None:
            trainer.fit(pib_model, loader, loader_val, ckpt_path=args.model.from_checkpoint)
        else:
            trainer.fit(pib_model, loader, loader_val)
    
    elif args.mode == 'validate':
        pib_model = get_model(args.model.name, **args.model)

        dataset_val = PIBDataset(mode=args.mode, data_path=args.data.data_path, train_validate_test_split=args.data.train_validate_test_split)
        loader_val = DataLoader(dataset_val, batch_size=args.data.validate_batch_size, shuffle=False)
        if args.model.from_checkpoint is None:
            raise ValueError('Checkpoint file path not provided for validation mode')
        trainer = L.Trainer(default_root_dir=args.train.root_dir)
        trainer.validate(pib_model, loader_val, ckpt_path=args.model.from_checkpoint)

    elif args.mode == 'test':
        raise NotImplementedError('Test mode not implemented yet')
    
    elif args.mode == 'validate_save':
        pib_model = get_model(args.model.name, **args.model)

        dataset_val = PIBDataset(mode='validate', data_path=args.data.data_path, train_validate_test_split=args.data.train_validate_test_split)
        loader_val = DataLoader(dataset_val, batch_size=args.data.validate_batch_size, shuffle=False)
        if args.model.from_checkpoint is None:
            raise ValueError('Checkpoint file path not provided for validate_save mode')
        trainer = L.Trainer(default_root_dir=args.train.root_dir, enable_checkpointing=False, logger=False)
        returned_values = trainer.predict(pib_model, loader_val, ckpt_path=args.model.from_checkpoint, return_predictions=True)
        outputs = [returned_value[0] for returned_value in returned_values]
        labels = [returned_value[1] for returned_value in returned_values]
        model_llrs = [returned_value[2] for returned_value in returned_values]
        if args.save.plot_path:
            import matplotlib.pyplot as plt
            i = 0
            '''
            while i < len(outputs[0]):
                if labels[0][i][0] == 1:
                    i += 1
                else:
                    break
            '''
            plt.plot(model_llrs[0][i], label='Model LLR')
            plt.plot(labels[0][i], label='original')
            plt.legend(fontsize='x-large')
            plt.grid()
            plt.xlabel('Time', fontsize='x-large')
            plt.ylabel('Prediction', fontsize='x-large')
            plt.savefig(args.save.plot_path)

    else:
        raise ValueError(f'Invalid mode {args.mode}')