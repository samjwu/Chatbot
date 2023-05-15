"""
Decode variable-length input sequence and fixed-length context vector 
to variable-length output sequence.
Recurrent Neural Networks (RNN) that takes an input word and fixed-length context vector
and returns a guess for the next word in the sequence and a hidden state to use in the next iteration.
"""

import torch.nn


class Decoder(torch.nn.Module):
    def __init__(
        self,
        attention_model: str,
        embedding: torch.nn.Embedding,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super(Decoder, self).__init__()

        self.attention_model = attention_model
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru = torch.nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(0 if num_layers == 1 else dropout),
        )
        self.embedding_dropout = torch.nn.Dropout(dropout)
        self.concat = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.out = torch.nn.Linear(hidden_size, output_size)
