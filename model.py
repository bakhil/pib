import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import scipy.signal as signal

# Use @register_model decorator to register the model.
# Model can then be obtained using get_model(model_name, **kwargs)
# with model_name from the config file.
_model_list = {}
def register_model(cls):
    _model_list[cls.__name__] = cls
    return cls
def get_model(model_name, **kwargs):
    return _model_list[model_name](**kwargs)

@register_model
class PIBTransformer(utils.PIBMainModel):
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
        accel_reshaped = torch.einsum('nlc->ncl', accel)
        accel_normalized = torch.einsum('ncl->nlc', self.normalize_input(accel_reshaped))
        accel_projected_reshaped = torch.einsum('nlc->ncl', self.input_projection(accel_normalized))
        x = torch.einsum('ncl->nlc', self.normalize_projection(accel_projected_reshaped))

        # Add positional encoding
        pos = torch.arange(x.shape[1]-1, -1, -1, device=x.device)[:, None]
        den = torch.exp(-torch.arange(0, x.shape[2], 2, device=x.device)[None, :] * torch.log(torch.tensor([10000])).item() / x.shape[2]) 
        pos_enc = torch.zeros(x.shape[1], x.shape[2], device=x.device)
        pos_enc[:, 0::2] = torch.sin(pos * den)
        pos_enc[:, 1::2] = torch.cos(pos * den)
        pos_enc = pos_enc[None, :, :]
        x = x + pos_enc

        transformer_output = self.transformer(x, mask=nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device), is_causal=True)
        return self.output_projection(transformer_output)
        
@register_model
class PIBFilTransformer(utils.PIBMainModel):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, fil_size,
                 cutoff, fs, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        # self.initial_filter = nn.Conv1d(3, d_model, fil_size, padding=fil_size-1)
        self.fil_size = fil_size
        self.cutoff = cutoff
        self.fs = fs
        # self.normalize_input = nn.BatchNorm1d(3)
        self.input_projection = nn.Linear(3, d_model)
        self.normalize_projection = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_projection = nn.Linear(d_model, 2)

    def forward(self, accel, ts=None):
        accel_normalized = (accel - torch.mean(accel, dim=1, keepdim=True)) / torch.std(accel, dim=1, keepdim=True)
        accel_reshaped = torch.einsum('nlc->ncl', accel_normalized)
        filter_taps = torch.tensor(signal.firwin(self.fil_size, cutoff=self.cutoff, fs=self.fs), device=accel.device, dtype=torch.float)[None, None, :].expand(3, 1, self.fil_size)
        accel_filtered = F.conv1d(accel_reshaped, weight=filter_taps, padding=self.fil_size-1, groups=3)[..., :accel.shape[1]]
        # accel_filtered = self.initial_filter(accel_reshaped)[:, :, :accel.shape[1]]
        # accel_normalized = torch.einsum('ncl->nlc', self.normalize_input(accel_reshaped))
        # accel_projected_reshaped = torch.einsum('nlc->ncl', self.input_projection(accel_normalized))
        # x = torch.einsum('ncl->nlc', self.normalize_projection(accel_projected_reshaped))
        accel_filtered_nlc = torch.einsum('ncl->nlc', accel_filtered)
        if self.fs/(4.0*self.cutoff) > 2.:
            sample_frac = int(self.fs/(4.0*self.cutoff))
        else:
            sample_frac = 1
        input_batch = torch.cat([accel_filtered_nlc[:, i::sample_frac, :] for i in range(sample_frac)], dim=0)
        x = self.normalize_projection(self.input_projection(input_batch))

        # Add positional encoding
        pos = torch.arange(x.shape[1]-1, -1, -1, device=x.device)[:, None]
        den = torch.exp(-torch.arange(0, x.shape[2], 2, device=x.device)[None, :] * torch.log(torch.tensor([10000])).item() / x.shape[2]) 
        pos_enc = torch.zeros(x.shape[1], x.shape[2], device=x.device)
        pos_enc[:, 0::2] = torch.sin(pos * den)
        pos_enc[:, 1::2] = torch.cos(pos * den)
        pos_enc = pos_enc[None, :, :]
        x = x + pos_enc

        transformer_output = self.transformer(x, mask=nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device), is_causal=True)
        downsampled_output = self.output_projection(transformer_output)
        output = torch.zeros(accel.shape[0], accel.shape[1], 2, device=accel.device)
        for i in range(sample_frac):
            output[:, i::sample_frac, :] = downsampled_output[i*accel.shape[0]:(i+1)*accel.shape[0], :, :]
        return output
        