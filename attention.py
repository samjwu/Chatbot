"""
Attention is to enhance certain parts of the input sequence and diminish other parts.
Calculated using output vectors from the encoder and hidden state from the decoder.
"""

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
