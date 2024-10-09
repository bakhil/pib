import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
import copy

class PIBMainModel(L.LightningModule):
    def __init__(self, ignore_initial: int = 100, ema_decay: float = 0.999):
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

    def on_before_zero_grad(self, *args, **kwargs):
        if self.ema_copy is None:
            self.ema_copy = torch.optim.swa_utils.AveragedModel(self, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(self.ema_decay))
        self.ema_copy.update_parameters(self)


# Use @register decorator to register the model.
# Model can then be obtained using get_model(model_name, **kwargs)
# with model_name from the config file.
_model_list = {}
def register(cls):
    _model_list[cls.__name__] = cls
    return cls
def get_model(model_name, **kwargs):
    return _model_list[model_name](**kwargs)

@register
class PIBTransformer(PIBMainModel):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.normalize_input = nn.BatchNorm1d(3)
        self.input_projection = nn.Linear(3, d_model)
        self.normalize_projection = nn.BatchNorm1d(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_projection = nn.Linear(d_model, 2)

    def forward(self, accel, ts=None):
        x = self.normalize_projection(self.input_projection(self.normalize_input(accel)))

        # Add positional encoding
        pos = torch.arange(x.shape[1], device=x.device)[:, None]
        den = torch.exp(-torch.arange(x.shape[2], device=x.device)[None, :] * torch.log(torch.tensor([10000])).item() / x.shape[2])
        pos_enc = torch.zeros(x.shape[1], x.shape[2], device=x.device)
        pos_enc[:, 0::2] = torch.sin(pos * den)
        pos_enc[:, 1::2] = torch.cos(pos * den)
        pos_enc = pos_enc[None, :, :]
        x = x + pos_enc

        transformer_output = self.transformer(x, mask=nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device))
        return self.output_projection(transformer_output)
        