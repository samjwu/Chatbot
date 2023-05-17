"""
Encode variable-length input sequence to fixed-length context vector.
Use Recurrent Neural Networks (RNN) and Gated Recurrent Unit (GRU)
to yield an output vector and hidden state vector each step.
"""

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

    def forward_pass(
        self,
        input_sentence: Tensor,
        input_lengths: Tensor,
        hidden_state_vectors: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Iterate through input sentence one word at a time.
        Yield an output vector and hidden state vector each step.
        """
        # convert word indices to embeddings
        # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        embedded = self.embedding(input_sentence)
        # pack a Tensor containing padded sequences of variable length
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # do a forward pass through the GRU
        # passing the hidden state vector to the next time step and recording the output vector
        output_vectors, hidden_state_vectors = self.gru(packed, hidden_state_vectors)
        # pad a packed batch of variable length sequences (inverse of pack_padded_sequence)
        output_vectors, _ = torch.nn.utils.rnn.pad_packed_sequence(output_vectors)
        # sum the output vectors
        output_vectors = (
            output_vectors[:, :, : self.hidden_size]
            + output_vectors[:, :, self.hidden_size :]
        )

        return output_vectors, hidden_state_vectors
