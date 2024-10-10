import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

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
        