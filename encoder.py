"""
Encode variable-length input sequence to fixed-length context vector.
Use Recurrent Neural Networks (RNN) and Gated Recurrent Unit (GRU)
to yield an output vector and hidden state vector each step.
"""

import torch
import torch.nn


class Encoder(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        embedding: torch.nn.Embedding,
        num_layers: int = 1,
        dropout: int = 0,
    ) -> None:
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = embedding
        self.num_layers = num_layers

        self.gru = torch.nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(0 if num_layers == 1 else dropout),
            bidirectional=True,
        )

    def forward(
        self,
        input_sentence: torch.Tensor,
        input_lengths: torch.Tensor,
        hidden_state_vector: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Iterate through input sentence one word at a time.
        Yield an output vector and hidden state vector each step.
        """
        # convert word indices to embeddings
        # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        embedded = self.embedding(input_sentence)

        # pack a tensor containing padded sequences of variable length
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        # do a forward pass through the GRU
        # passing the hidden state vector to the next time step and recording the output vector
        output_vector, hidden_state_vector = self.gru(packed, hidden_state_vector)

        # pad a packed batch of variable length sequences (inverse of pack_padded_sequence)
        output_vector, _ = torch.nn.utils.rnn.pad_packed_sequence(output_vector)

        # get the sums the output vector
        output_vector = (
            output_vector[:, :, : self.hidden_size]
            + output_vector[:, :, self.hidden_size :]
        )

        return output_vector, hidden_state_vector
