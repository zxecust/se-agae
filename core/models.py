"""网络模型模块：定义图自编码器和工况序列到潜变量的 LSTM 预测器"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv, global_add_pool, global_mean_pool
from torch_geometric.utils import softmax


class GNNEncoder(nn.Module):
    """图编码器，通过节点特征提取与图内汇总，将每帧物理场压缩为潜变量"""

    def __init__(self, pos_dim, config):
        super().__init__()
        # GNNEncoder 的配置来自外部读取的数据，结构为字典
        hidden = config["hidden_dim"]
        # 是否在编码器中使用节点坐标作为附加特
        self.use_pos = config["use_pos"]
        # 是否在图内汇总时使用注意力机制，决定了 readout_gate 的存在与否
        self.attention_pool = config["attention_pool"]
        # config 为 model 子配置，pos_dim 为坐标维度 D，节点场值为单通道
        in_dim = 1 + (pos_dim if self.use_pos else 0)
        # ModuleList 注册图卷积层，使参数可由优化器训练并写入 state_dict
        self.convs = nn.ModuleList([SAGEConv(in_dim, hidden)])
        # 首层将输入映射到 hidden，后续层保持节点隐藏特征宽度
        self.convs.extend(SAGEConv(hidden, hidden) for _ in range(config["encoder_layers"] - 1))
        # 图内汇总后再映射到 latent_dim，末层线性输出允许潜变量取正负值
        self.fc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                nn.Linear(hidden, config["latent_dim"]))
        # 注意力读出时，为每个节点生成一个待归一化的重要性分数
        self.readout_gate = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, pos, edge_index, batch):
        """将 B 帧节点场值 [B*N,1] 编码为逐帧潜变量 [B,L]"""
        # 1. 组合节点输入，并沿各帧独立的图连接提取特征
        # pos [B*N,D]、edge_index [2,B*E]，batch [B*N] 标记节点所属帧
        h = torch.cat([x, pos], dim=1) if self.use_pos else x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))

        # 2. 按 batch 分组，将节点特征汇总成每帧一个向量 [B,hidden]
        if self.attention_pool:
            # softmax 在每帧图内部归一化，使该帧全部节点的权重之和为 1
            weights = softmax(self.readout_gate(h), batch)
            pooled = global_add_pool(weights * h, batch)
        else:
            pooled = global_mean_pool(h, batch)

        # 3. 将每帧汇总特征映射到潜空间，L 为 latent_dim
        return self.fc(pooled)


class GNNDecoder(nn.Module):
    """图解码器，将逐帧潜变量与节点坐标结合，重构单通道节点物理场"""

    def __init__(self, pos_dim, config):
        super().__init__()
        hidden = config["hidden_dim"]
        # 解码器始终使用坐标，use_pos 只控制编码器是否拼接坐标
        self.convs = nn.ModuleList([SAGEConv(pos_dim + config["latent_dim"], hidden)])
        # 后续层保持节点隐藏特征宽度，末层线性输出允许场值取正负值
        self.convs.extend(SAGEConv(hidden, hidden) for _ in range(config["decoder_layers"] - 1))
        # 每个节点输出一个场值，末层不限制符号，以适配归一化后的物理场
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z, pos, edge_index, batch):
        """使用潜变量 z [B,L] 和对应批图，返回重构场 [B*N,1]"""
        # z[batch] 将每帧潜变量分配到该帧的所有节点，拼接后为 [B*N,D+L]
        h = torch.cat([pos, z[batch]], dim=1)
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        return self.head(h)


class Autoencoder(nn.Module):
    """图自编码器，串联编码器和解码器，学习单帧物理场的压缩与重构"""

    def __init__(self, pos_dim, config):
        super().__init__()
        self.encoder = GNNEncoder(pos_dim, config)
        self.decoder = GNNDecoder(pos_dim, config)

    def forward(self, x, pos, edge_index, batch):
        """输入与输出均为 [B*N,1]，归一化和反归一化由调用方处理"""
        # 编码为 [B,L] 后用同一批图重构；需要潜变量时可单独调用 encoder
        z = self.encoder(x, pos, edge_index, batch)
        return self.decoder(z, pos, edge_index, batch)


class LSTMLatentPredictor(nn.Module):
    """序列预测器，将完整工况的特征序列映射到 AE 原始潜空间的逐帧潜变量"""

    def __init__(self, in_dim, latent_dim, config):
        super().__init__()
        # config 为 predictor 子配置，in_dim 为特征数 F，latent_dim 为输出维度 L
        # batch_first=True 约定输入为 [B,T,F]，此处 B 为工况序列数
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=config["hidden_size"],
            num_layers=config["num_layers"], batch_first=True,
            # dropout 用于堆叠 LSTM 层之间，单层时关闭
            dropout=config["dropout"] if config["num_layers"] > 1 else 0.0,
        )
        # 逐帧将隐藏状态映射到潜变量维度，损失中的标准化不改变此输出定义
        self.head = nn.Linear(config["hidden_size"], latent_dim)

    def forward(self, sequence, lengths=None):
        """预测工况特征 [B,T,F] 对应的潜变量，lengths 可排除补齐帧"""
        # 未传入初始隐藏状态，每次调用都从零状态开始，不跨工况延续状态
        if lengths is None:
            output, _ = self.lstm(sequence)
        else:
            # 长度保留在 CPU；允许输入工况不按长度排序，打包后跳过补齐位置
            packed = nn.utils.rnn.pack_padded_sequence(
                sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
            output, _ = self.lstm(packed)
            # 解包恢复输入工况顺序，时间维为本批最大有效长度
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # 逐帧输出 [B,Tout,L]；线性头可使补齐位置非零，损失仍须用 mask 排除
        return self.head(output)
