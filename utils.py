import torch
import lightning as L
from lightning.pytorch.cli import LightningArgumentParser
import torch.nn.functional as F

# Common functions for all models
# Models in model.py can then inherit from this class
class PIBMainModel(L.LightningModule):
    def __init__(self, ignore_initial: int = 100, ema_decay: float = 0.999, **kwargs):
        super().__init__()
        self.ignore_initial = ignore_initial
        self.save_hyperparameters()
        self.ema_copy = None
        self.ema_decay = ema_decay

    def training_step(self, batch, batch_idx):
        accel, ts, labels = batch
        output = self(accel, ts)
        loss = F.cross_entropy(output[:, self.ignore_initial:], labels[:, self.ignore_initial:])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        accel, ts, labels = batch
        model_output = self.ema_copy(accel, ts)
        output = torch.where(model_output[..., 0] > model_output[..., 1], 0, 1)
        correct = torch.sum(output[:, self.ignore_initial:] == labels[:, self.ignore_initial:], dim=1) / (output.shape[1] - self.ignore_initial)
        avg_correct = torch.mean(correct)
        pos_pred_avg = torch.mean(correct[correct > 0.5])
        neg_pred_avg = torch.mean(correct[correct <= 0.5])
        self.log('val_acc', avg_correct)
        self.log('val_pos_pred_avg', pos_pred_avg)
        self.log('val_neg_pred_avg', neg_pred_avg)

    def on_train_batch_end(self, *args, **kwargs):
        if self.ema_copy is None:
            self.ema_copy = torch.optim.swa_utils.AveragedModel(self, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(self.ema_decay))
        self.ema_copy.update_parameters(self)

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
    parser.add_argument('--model.fresh_init',       help='initialize the model randomly (Optional)', action='store_true')

    # Training arguments
    parser.add_argument('--train.max_epochs',      help='max number of epochs', type=int)
    parser.add_argument('--train.root_dir',        help='root directory for the dataset')
    parser.add_argument('--train.batch_size',      help='batch size', type=int)
    parser.add_argument('--train.optimizer',       help='optimizer')
    parser.add_argument('--train.lr',              help='learning rate', type=float)

    # Data arguments
    parser.add_argument('--data.data_path',                 help='json file path for the dataset')
    parser.add_argument('--data.train_validate_test_split', help='train/validate/test ratio as integers', type=list[int])
    parser.add_argument('--data.train_seq_len',             help='sequence length for training data', type=int)

    return parser