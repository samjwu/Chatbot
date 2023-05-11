"""
Encode variable-length input sequence to fixed-length context vector.
Use Recurrent Neural Networks (RNN) and Gated Recurrent Unit (GRU)
to yield an output vector and hidden state vector each step.
"""

import torch.nn


class Encoder(torch.nn.Module):
    def __init__(self, hidden_size: int, embedding: torch.nn.Embedding, num_layers: int=1, dropout: int=0) -> None:
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = torch.nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(0 if num_layers == 1 else dropout),
            bidirectional=True
        )
