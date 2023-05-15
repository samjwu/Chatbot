"""
Attention is to enhance certain parts of the input sequence and diminish other parts.
Calculated using output vectors from the encoder and hidden state from the decoder.
"""

import torch
import torch.nn


class Attention(nn.Module):
    def __init__(self, method: str, hidden_size: int) -> None:
        super(Attention, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")

        if self.method == 'general':
            self.attention = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attention = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.parameter = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def calculate_dot_score(self, hidden_state_vectors, encoder_output) -> Tensor:
        return torch.sum(hidden_state_vectors * encoder_output, dim=2)

    def calculate_general_score(self, hidden_state_vectors, encoder_output) -> Tensor:
        energy = self.attention(encoder_output)
        return torch.sum(hidden_state_vectors * energy, dim=2)

    def calculate_concat_score(self, hidden_state_vectors, encoder_output) -> Tensor:
        energy = self.attention(torch.cat(
            (hidden_state_vectors.expand(encoder_output.size(0), -1, -1), encoder_output),
            2)).tanh()
        return torch.sum(self.parameter * energy, dim=2)
