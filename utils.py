import torch
import lightning as L
from lightning.pytorch.cli import LightningArgumentParser
import torch.nn.functional as F
import copy

# Common functions for all models
# Models in model.py can then inherit from this class
class PIBMainModel(L.LightningModule):
    def __init__(self, ignore_initial: int = 100, ema_decay: float = 0.999, 
                    optimizer: str = '', lr: float = 0.001,
                    train_seq_length: int = 1000, **kwargs):
        super().__init__()
        self.ignore_initial = ignore_initial
        self.ema_copy = None
        self.ema_decay = ema_decay
        self.optimizer = optimizer
        self.lr = lr
        self.train_seq_length = train_seq_length
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        accel, ts, labels = batch
        output = self(accel, ts)
        loss = F.cross_entropy(output[:, self.ignore_initial:].reshape(-1, 2), labels[:, self.ignore_initial:].reshape(-1))
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        if self.ema_copy is None:
            # self.ema_copy = torch.optim.swa_utils.AveragedModel(self, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(self.ema_decay))
            self.ema_copy = copy.deepcopy(self)
            for p in self.ema_copy.parameters():
                p.requires_grad = False
        accel, ts, labels = batch
        model_output = torch.zeros(accel.shape[0], accel.shape[1], 2, device=accel.device, requires_grad=False)
        for i in range(self.train_seq_length-1, accel.shape[1]):
            start_index = max(0, i-self.train_seq_length+1)
            model_output[:, i, :] = self.ema_copy(accel[:, start_index:i+1], ts[:, start_index:i+1])[:, -1, :].detach()
        model_llr = model_output[..., 1] - model_output[..., 0]
        output = torch.where(model_llr > 0, 1, 0)
        output_seg = torch.where(torch.mean(model_llr[:, self.train_seq_length-1:], dim=1) > 0, 1, 0)[:, None]

        correct = torch.mean((output[:, self.train_seq_length-1:] == labels[:, self.train_seq_length-1:])*1.0)
        correct_seg = torch.mean((labels == output_seg)*1.0)
        
        self.log('val_acc', correct.item())
        self.log('val_acc_segmented', correct_seg.item())
        self.log('val_abs_llr', torch.mean(torch.abs(model_llr[:, self.train_seq_length-1:])).item())

    def test_step(self, batch, batch_idx):
        if self.ema_copy is None:
            # self.ema_copy = torch.optim.swa_utils.AveragedModel(self, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(self.ema_decay))
            self.ema_copy = copy.deepcopy(self)
            for p in self.ema_copy.parameters():
                p.requires_grad = False
        accel, ts, labels = batch
        model_output = torch.zeros(accel.shape[0], accel.shape[1], 2, device=accel.device, requires_grad=False)
        with torch.no_grad():
            model_output[:, :self.train_seq_length, :] = self.ema_copy(accel[:, :self.train_seq_length], ts[:, :self.train_seq_length])
            for i in range(self.train_seq_length, accel.shape[1]):
                model_output[:, i, :] = self.ema_copy(accel[:, i-self.train_seq_length+1:i+1], ts[:, i-self.train_seq_length+1:i+1])[:, -1, :].detach()
        model_llr = model_output[..., 1] - model_output[..., 0]
        output = torch.where(model_llr > 0, 1, 0)
        output_seg = torch.where(torch.mean(model_llr[:, self.train_seq_length-1:], dim=1) > 0, 1, 0)[:, None]

        correct = torch.mean((output[:, self.train_seq_length-1:] == labels[:, self.train_seq_length-1:])*1.0)
        correct_seg = torch.mean((labels == output_seg)*1.0)

        self.log('test_acc', correct.item())
        self.log('test_acc_segmented', correct_seg.item())

    def configure_optimizers(self):
        if self.ema_copy is not None:
            for p in self.ema_copy.parameters():
                p.requires_grad = False
        optimizer = eval('torch.optim.' + self.optimizer)
        return optimizer(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)

    def on_train_batch_end(self, *args, **kwargs):
        if self.ema_copy is None:
            # self.ema_copy = torch.optim.swa_utils.AveragedModel(self, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(self.ema_decay))
            self.ema_copy = copy.deepcopy(self)
            for p in self.ema_copy.parameters():
                p.requires_grad = False
        # self.ema_copy.update_parameters(self)
        model_state_dict = self.state_dict()
        for param_name, ema_param in self.ema_copy.state_dict().items():
            model_param = model_state_dict[param_name]
            if torch.is_floating_point(ema_param):
                ema_param.lerp_(model_param, 1.-self.ema_decay)
                ema_param.requires_grad = False
            else:
                ema_param.copy_(model_param)

    # This is needed for ema_copy to be loaded
    # Gives error otherwise
    def load_state_dict(self, *args, **kwargs):
        if self.ema_copy is None:
            # self.ema_copy = torch.optim.swa_utils.AveragedModel(self, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(self.ema_decay))
            self.ema_copy = copy.deepcopy(self)
            for p in self.ema_copy.parameters():
                p.requires_grad = False
        super().load_state_dict(*args, **kwargs)

# Argument parser for the main script
def get_parser():
    parser = LightningArgumentParser()

    # Main script arguments
    parser.add_argument('--config',             help='config file path', action='config', required=True)
    parser.add_argument('--task',               help='segmented or streaming', choices=['segmented', 'streaming'], default='segmented')
    parser.add_argument('--mode',               help='train/validate/test', choices=['train', 'validate', 'test'], required=True)
    parser.add_argument('--lightning_seed',     help='random seed for lightning', type=int)

    # Model arguments
    parser.add_argument('--model.name',             help='model name', required=True)
    parser.add_argument('--model.d_model',          help='model d_model', type=int)
    parser.add_argument('--model.nhead',            help='number of heads', type=int)
    parser.add_argument('--model.num_layers',       help='number of layers', type=int)
    parser.add_argument('--model.dim_feedforward',  help='feedforward dimension', type=int)
    parser.add_argument('--model.dropout',          help='dropout rate', type=float)
    parser.add_argument('--model.ignore_initial',   help='initial samples to ignore during training', type=int)
    parser.add_argument('--model.ema_decay',        help='ema decay rate', type=float)
    parser.add_argument('--model.from_checkpoint',  help='checkpoint file path (Optional)')
    parser.add_argument('--model.optimizer',        help='optimizer will be torch.optim.<model.optimizer>', type=str)
    parser.add_argument('--model.lr',               help='learning rate', type=float)
    parser.add_argument('--model.train_seq_len',    help='sequence length for training', type=int)


    # Training arguments
    parser.add_argument('--train.max_epochs',               help='max number of epochs', type=int)
    parser.add_argument('--train.root_dir',                 help='root directory for the dataset')
    parser.add_argument('--train.accumulate_grad_batches',  help='grad accumulate batches', type=int, default=1)

    # Data arguments
    parser.add_argument('--data.data_path',                 help='json file path for the dataset')
    parser.add_argument('--data.train_validate_test_split', help='train/validate/test ratio as integers', type=list[int])
    parser.add_argument('--data.train_batch_size',          help='batch size for training', type=int)
    parser.add_argument('--data.validate_batch_size',       help='batch size for validation', type=int)

    return parser