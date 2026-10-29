"""网络模型模块：定义图自编码器和工况序列到潜变量的 LSTM 预测器"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv, global_add_pool, global_mean_pool
from torch_geometric.utils import softmax


class GNNEncoder(nn.Module):
    def __init__(self, pos_dim, config):
        super().__init__()
        hidden = config["hidden_dim"]
        self.use_pos = config["use_pos"]
        self.attention_pool = config["attention_pool"]
        in_dim = 1 + (pos_dim if self.use_pos else 0)
        self.convs = nn.ModuleList([SAGEConv(in_dim, hidden)])
        self.convs.extend(SAGEConv(hidden, hidden) for _ in range(config["encoder_layers"] - 1))
        self.fc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                nn.Linear(hidden, config["latent_dim"]))
        self.readout_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, pos, edge_index, batch):
        h = torch.cat([x, pos], dim=1) if self.use_pos else x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        if self.attention_pool:
            weights = softmax(self.readout_gate(h), batch)
            pooled = global_add_pool(weights * h, batch)
        else:
            pooled = global_mean_pool(h, batch)
        return self.fc(pooled)


class GNNDecoder(nn.Module):
    def __init__(self, pos_dim, config):
        super().__init__()
        hidden = config["hidden_dim"]
        self.convs = nn.ModuleList([SAGEConv(pos_dim + config["latent_dim"], hidden)])
        self.convs.extend(SAGEConv(hidden, hidden) for _ in range(config["decoder_layers"] - 1))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z, pos, edge_index, batch):
        h = torch.cat([pos, z[batch]], dim=1)
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        return self.head(h)


class Autoencoder(nn.Module):
    def __init__(self, pos_dim, config):
        super().__init__()
        self.encoder = GNNEncoder(pos_dim, config)
        self.decoder = GNNDecoder(pos_dim, config)

    def forward(self, x, pos, edge_index, batch):
        z = self.encoder(x, pos, edge_index, batch)
        return self.decoder(z, pos, edge_index, batch)


class LSTMLatentPredictor(nn.Module):
    def __init__(self, in_dim, latent_dim, config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=config["hidden_size"],
            num_layers=config["num_layers"], batch_first=True,
            dropout=config["dropout"] if config["num_layers"] > 1 else 0.0,
        )
        self.head = nn.Linear(config["hidden_size"], latent_dim)

    def forward(self, sequence, lengths=None):
        if lengths is None:
            output, _ = self.lstm(sequence)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(
                sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
            output, _ = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return self.head(output)
