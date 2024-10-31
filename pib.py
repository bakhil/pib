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
        
        dataset = PIBDataset(mode=args.mode, data_path=args.data.data_path, 
                             train_validate_test_split=args.data.train_validate_test_split,
                             train_seq_len=args.model.train_seq_len)
        loader = DataLoader(dataset, batch_size=args.data.train_batch_size, shuffle=True)
        dataset_val = PIBDataset(mode='validate', data_path=args.data.data_path, 
                                 train_validate_test_split=args.data.train_validate_test_split,
                                 train_seq_len=args.model.train_seq_len)
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

        dataset_val = PIBDataset(mode=args.mode, data_path=args.data.data_path, 
                                 train_validate_test_split=args.data.train_validate_test_split,
                                 train_seq_len=args.model.train_seq_len)
        loader_val = DataLoader(dataset_val, batch_size=args.data.validate_batch_size, shuffle=False)
        if args.model.from_checkpoint is None:
            raise ValueError('Checkpoint file path not provided for validation mode')
        trainer = L.Trainer(default_root_dir=args.train.root_dir, enable_checkpointing=False, logger=False)
        trainer.validate(pib_model, loader_val, ckpt_path=args.model.from_checkpoint)

    elif args.mode == 'test':
        raise NotImplementedError('Test mode not implemented yet')
    
    elif args.mode == 'validate_save':
        pib_model = get_model(args.model.name, **args.model)

        dataset_val = PIBDataset(mode='validate', data_path=args.data.data_path, 
                                 train_validate_test_split=args.data.train_validate_test_split,
                                 train_seq_len=args.model.train_seq_len)
        loader_val = DataLoader(dataset_val, batch_size=args.data.validate_batch_size, shuffle=False)
        if args.model.from_checkpoint is None:
            raise ValueError('Checkpoint file path not provided for validate_save mode')
        trainer = L.Trainer(default_root_dir=args.train.root_dir, enable_checkpointing=False, logger=False)
        returned_values = trainer.predict(pib_model, loader_val, ckpt_path=args.model.from_checkpoint, return_predictions=True)
        outputs = torch.cat([returned_value[0] for returned_value in returned_values], dim=0)
        labels = torch.cat([returned_value[1] for returned_value in returned_values], dim=0)
        model_llrs = torch.cat([returned_value[2] for returned_value in returned_values], dim=0)
        ts_list = torch.cat([returned_value[3] for returned_value in returned_values], dim=0)
        prev_sums = torch.cat([returned_value[4] for returned_value in returned_values], dim=0)

        avg_latency_list = []
        latency_lists = []
        for i in range(len(outputs)):
            transitions = torch.arange(1, len(outputs[i]), device=outputs.device, dtype=torch.long)[(labels[i][1:] - labels[i][:-1]) != 0]
            latency_list = []
            for j in transitions:
                if ts_list[i][j] < 0:
                    break
                k = int(j + 0)   # k gets assigned to j's pointer and changes j otherwise
                while k < len(outputs[i]):
                    if torch.mean((outputs[i][k:k+15*250+1] == labels[i][j])*1.0) == 1.:
                        break
                    k += 1
                latency_list.append((k - j) / 250.)
            if len(latency_list) > 0:
                avg_latency_list.append(torch.mean(torch.tensor(latency_list)))
                latency_lists.append(latency_list)
        accuracy = torch.mean((outputs == labels)[ts_list > 0]*1.0)
        print('#####################################################')
        print(f'Average latency: {torch.mean(torch.tensor(avg_latency_list)):.3e} s')
        print(f'Std of latency: {torch.std(torch.tensor(avg_latency_list)):.3e} s')
        print(f'Average accuracy: {accuracy:.3f}')
        print('#####################################################')
        if args.save.plot_path:
            import matplotlib.pyplot as plt
            i = torch.randint(0, len(outputs), (1,)).item()
            '''
            while i < len(outputs[0]):
                if labels[0][i][0] == 1:
                    i += 1
                else:
                    break
            '''
            # plt.plot(model_llrs[0][i], label='Model LLR')
            fig, axs = plt.subplots(3, 1, squeeze=False, figsize=(10, 15))
            axs[0, 0].plot(torch.arange(len(outputs[i][ts_list[i] >= 0.]))/250., outputs[i][ts_list[i] >= 0.], label='outputs')
            axs[0, 0].plot(torch.arange(len(labels[i][ts_list[i] >= 0.]))/250., labels[i][ts_list[i] >= 0.], label='original')
            axs[0, 0].legend(fontsize='x-large')
            axs[0, 0].grid()
            axs[0, 0].set_xlabel('Time (s)', fontsize='x-large')
            axs[0, 0].set_ylabel('Prediction', fontsize='x-large')
            axs[0, 0].set_title(f'Sample index {i}', fontsize='x-large')

            axs[1, 0].plot(torch.arange(len(model_llrs[i][ts_list[i] >= 0.]))/250., model_llrs[i][ts_list[i] >= 0.], label='model llr')
            axs[1, 0].plot(torch.arange(len(labels[i][ts_list[i] >= 0.]))/250., labels[i][ts_list[i] >= 0.], label='original')
            axs[1, 0].legend(fontsize='x-large')
            axs[1, 0].grid()
            axs[1, 0].set_xlabel('Time (s)', fontsize='x-large')
            axs[1, 0].set_ylabel('Value', fontsize='x-large')

            axs[2, 0].plot(torch.arange(len(model_llrs[i][ts_list[i] >= 0.]))/250., prev_sums[i][ts_list[i] >= 0.], label='Avg sums')
            axs[2, 0].plot(torch.arange(len(labels[i][ts_list[i] >= 0.]))/250., labels[i][ts_list[i] >= 0.], label='original')
            axs[2, 0].legend(fontsize='x-large')
            axs[2, 0].grid()
            axs[2, 0].set_xlabel('Time (s)', fontsize='x-large')
            axs[2, 0].set_ylabel('Value', fontsize='x-large')
            plt.savefig(args.save.plot_path)

    else:
        raise ValueError(f'Invalid mode {args.mode}')