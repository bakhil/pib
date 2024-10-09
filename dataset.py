import torch
from torch.utils.data import Dataset
import json

class PIBDatasetMain(Dataset):
    '''
    Main Dataset class for the PIB dataset.
    Train/Validate/Test datasets can all pull from the main object 
    once it's contructed.
    '''
    def __init__(self, data_path: str):
        '''
        Args:
            data_path (str): path to the (json) data file
        '''
        super().__init__()
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.num_samples = len(data)
        self.seq_lens = [len(data[i]['ts']) for i in range(self.num_samples)]
        self.max_len = max(self.seq_lens)

        # check if segmented problem or streaming problem based on labels
        if 'label' in data[0]:
            label_key = 'label'    # segmented
        elif 'labels' in data[0]:
            label_key = 'labels'   # streaming
        else:
            raise ValueError('Unknown data format!!')
        
        self.accel = torch.zeros(self.num_samples, self.max_len, 3)
        self.ts = torch.zeros(self.num_samples, self.max_len) - 1.
        self.labels = torch.zeros(self.num_samples, self.max_len, dtype=torch.long)
        for i in range(self.num_samples):
            self.accel[i, :self.seq_lens[i], :] = torch.tensor(data[i]['accel'])
            self.ts[i, :self.seq_lens[i]] = torch.tensor(data[i]['ts'])
            # For segmented problem, integer gets broadcasted to all timestamps
            self.labels[i, :self.seq_lens[i]] = torch.tensor(data[i][label_key])

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.accel[idx], self.ts[idx], self.labels[idx]

class PIBDataset(Dataset):
    '''
    Dataset class for the PIB dataset.
    Instantiates a PIBDatasetMain object and then splits into
    Train/Validate/Test data
    '''

    # Use this to avoid repetitive instantiation of the main dataset
    _main_instances = {}
    @classmethod
    def get_main_dataset(cls, data_path):
        if data_path not in cls._main_instances:
            cls._main_instances[data_path] = PIBDatasetMain(data_path)
        return cls._main_instances[data_path]
    
    def __init__(self, data_path: str, mode: str='train', train_seq_len: int=1000,
                   train_validate_test_split: list[int]=[14, 3, 3]):
        '''
        Args:
            data_path (str): path to the (json) data file
            mode (str): 'train' (default), 'validate', or 'test'
            train_seq_len (int): sequence length for training data (default: 1000)
            train_validate_test_split (list[int]): split ratios for train/validate/test
                    (default: [14, 3, 3])
        '''
        super().__init__()
        self.main_dataset = PIBDataset.get_main_dataset(data_path)

        # List of sample indices based on whether dataset is train/validate/test
        self.index_list = []
        split_id = 0
        for i in range(self.main_dataset.num_samples):
            if split_id < train_validate_test_split[0]:
                if mode == 'train':
                    self.index_list.append(i)
            elif split_id < train_validate_test_split[0] + train_validate_test_split[1]:
                if mode == 'validate':
                    self.index_list.append(i)
            else:
                if mode == 'test':
                    self.index_list.append(i)
            split_id += 1
            split_id %= sum(train_validate_test_split)
        if mode == 'train':
            self.train_seq_len = train_seq_len
            self.num_samples = sum([self.main_dataset.seq_lens[i]-train_seq_len+1 for i in self.index_list])
            self.cumsum_samples = torch.cumsum(torch.tensor([0] + [self.main_dataset.seq_lens[i]-train_seq_len+1 for i in self.index_list]), 0)
        else:
            self.num_samples = len(self.index_list)
        self.mode = mode

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.mode == 'train':
            # Find the sample index and the timestamp index
            sample_idx = torch.searchsorted(self.cumsum_samples, idx, right=True)-1
            idx = idx - self.cumsum_samples[sample_idx]
            return (self.main_dataset.accel[self.index_list[sample_idx], idx:idx+self.train_seq_len, :],
                    self.main_dataset.ts[self.index_list[sample_idx], idx:idx+self.train_seq_len],
                    self.main_dataset.labels[self.index_list[sample_idx], idx:idx+self.train_seq_len])
        else:
            return self.main_dataset[self.index_list[idx]]

if __name__ == '__main__':
    import time
    from torch.utils.data import DataLoader

    data_path = 'data/icassp-person-in-bed-track-1/train.json'

    # Train dataset
    start_time = time.time()
    dataset = PIBDataset(data_path, mode='train')
    end_time = time.time()
    print(f'Time taken to create train dataset: {end_time-start_time:.4f} seconds')

    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    start_time = time.time()
    for i, (accel, ts, labels) in enumerate(loader):
        print(f'\rDone with batch {i+1}/{len(loader)} of train dataloader', end='')
    print()
    end_time = time.time()
    time_per_batch = (end_time-start_time)/(i+1)
    print(f'Time per batch: {time_per_batch:.4f} seconds for train dataloader')

    # Validate dataset
    start_time = time.time()
    dataset_val = PIBDataset(data_path, mode='validate')
    end_time = time.time()
    print(f'Time taken to create validate dataset: {end_time-start_time:.4f} seconds')

    loader_val = DataLoader(dataset_val, batch_size=512, shuffle=True)
    start_time = time.time()
    for i, (accel, ts, labels) in enumerate(loader_val):
        print(f'\rDone with batch {i+1}/{len(loader_val)} of val dataloader', end='')
    print()
    end_time = time.time()
    time_per_batch = (end_time-start_time)/(i+1)
    print(f'Time per batch: {time_per_batch:.4f} seconds for val dataloader')
    