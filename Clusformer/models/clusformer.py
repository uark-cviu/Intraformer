import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Clusformer(nn.Module):
    def __init__(self, 
                num_querries=80, 
                hidden_dim=256, 
                nheads=8,
                num_encoder_layers=6, 
                num_decoder_layers=6):
        super().__init__()

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # create position encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, 1)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(num_querries, hidden_dim)) # num_classes, E
        # self.linear_encoding = nn.Parameter(torch.rand(num_querries, hidden_dim))

    def forward(self, inputs):
        # construct positional encodings
        # inputs = self.embedding(inputs)
        bs = inputs.size(0)
        # inputs = self.pos_encoder(inputs) # N, S, E
        # inputs = inputs + self.linear_encoding
        inputs = inputs.permute(1, 0, 2) # S, N, E

        query_pos = self.query_pos.unsqueeze(1).repeat(1, bs, 1)

        # propagate through the transformer
        h = self.transformer(inputs, query_pos)
        h = h.transpose(0, 1)
        
        logit = self.linear_class(h)
        logit = logit.view(logit.size(0), -1)
        return logit


if __name__ == "__main__":
    num_querries=80
    hidden_dim=256
    nheads=8
    num_encoder_layers=6
    num_decoder_layers=6

    model = Clusformer(
        num_querries=num_querries, 
        hidden_dim=hidden_dim, 
        nheads=nheads, 
        num_encoder_layers=num_decoder_layers, 
        num_decoder_layers=num_decoder_layers
    )

    model = model.cuda()

    inputs = torch.rand(32, 80, 256).cuda()
    output = model(inputs)
    print(output.shape)