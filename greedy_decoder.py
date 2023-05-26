"""
Decoding method that is optimal on a single time-step level and does not use teacher forcing. 
For each time step, choose the word from decoder output vector with the highest softmax value. 
"""

import torch.nn

from decoder import Decoder
from encoder import Encoder


class GreedyDecoder(torch.nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super(GreedyDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
