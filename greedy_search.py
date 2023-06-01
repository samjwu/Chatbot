"""
Decoding method that is optimal on a single time-step level and does not use teacher forcing. 
For each time step, choose the word from decoder output vector with the highest softmax value. 
"""

import torch
import torch.nn

import vocabulary
from decoder import Decoder
from encoder import Encoder


class GreedySearch(torch.nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super(GreedySearch, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, input_sequence: Tensor, input_length: Tensor, max_length: int
    ) -> tuple[Tensor, Tensor]:
        """Perform multiple forward passes on a given input sequence using RNN decoder."""
        encoder_output_vector, encoder_hidden_state_vector = self.encoder(
            input_sequence, input_length
        )

        # initialize decoder's first input and output tensors
        decoder_hidden_state_vector = encoder_hidden_state_vector[: decoder.num_layers]
        decoder_input_vector = (
            torch.ones(1, 1, device=device, dtype=torch.long) * vocabulary.START
        )
        word_tokens = torch.zeros([0], device=device, dtype=torch.long)
        softmax_scores = torch.zeros([0], device=device)

        # decode one word at a time
        for _ in range(max_length):
            decoder_output, decoder_hidden_state_vector = self.decoder(
                decoder_input_vector, decoder_hidden_state_vector, encoder_output_vector
            )

            # get the most likely word token and its softmax score
            decoder_softmax_scores, decoder_input_vector = torch.max(
                decoder_output, dim=1
            )
            word_tokens = torch.cat((word_tokens, decoder_input_vector), dim=0)
            softmax_scores = torch.cat((softmax_scores, decoder_softmax_scores), dim=0)

            # set current token as next decoder input (add a dimension)
            decoder_input_vector = torch.unsqueeze(decoder_input_vector, 0)

        return word_tokens, softmax_scores
