"""
Decode variable-length input sequence and fixed-length context vector 
to variable-length output sequence.
Recurrent Neural Networks (RNN) that takes an input word and fixed-length context vector
and returns a guess for the next word in the sequence and a hidden state to use in the next iteration.
"""

import torch
import torch.nn
import torch.nn.functional

from attention import Attention


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

        self.attention = Attention(attention_model, hidden_size)

    def forward(
        self,
        input_step: torch.Tensor,
        last_hidden_layer: torch.Tensor,
        encoder_output_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Iterate through input one step/word at a time.
        Yield an output vector and hidden state vector each step.
        """
        # get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        # do a forward pass through the GRU
        rnn_output, hidden_state_vector = self.gru(embedded, last_hidden_layer)

        # calculate attention weights from the current GRU output
        attention_weights = self.attention(rnn_output, encoder_output_vector)

        # multiply attention weights by encoder outputs to get weighted sum context vector
        context_vector = attention_weights.bmm(encoder_output_vector.transpose(0, 1))

        # concatenate weighted context vector and GRU output
        rnn_output = rnn_output.squeeze(0)
        context_vector = context_vector.squeeze(1)
        concat_input = torch.cat((rnn_output, context_vector), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # predict next word
        output_vector = self.out(concat_output)
        output_vector = torch.nn.functional.softmax(output_vector, dim=1)

        return output_vector, hidden_state_vector
