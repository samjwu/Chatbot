"""
Attention is to enhance certain parts of the input sequence and diminish other parts.
Calculated using output vector from the encoder and hidden state vector from the decoder.
"""

import torch
import torch.nn
import torch.nn.functional


class Attention(torch.nn.Module):
    def __init__(self, method: str, hidden_size: int) -> None:
        super(Attention, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method not in ["dot", "general", "concat"]:
            raise ValueError(self.method, "is not an appropriate attention method.")

        if self.method == "general":
            self.attention = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == "concat":
            self.attention = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.parameter = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def calculate_dot_score(
        self, hidden_state_vector: torch.Tensor, encoder_output_vector: torch.Tensor
    ) -> torch.Tensor:
        """Reference: https://arxiv.org/pdf/1508.04025.pdf"""
        return torch.sum(hidden_state_vector * encoder_output_vector, dim=2)

    def calculate_general_score(
        self, hidden_state_vector: torch.Tensor, encoder_output_vector: torch.Tensor
    ) -> torch.Tensor:
        """Reference: https://arxiv.org/pdf/1508.04025.pdf"""
        energy = self.attention(encoder_output_vector)
        return torch.sum(hidden_state_vector * energy, dim=2)

    def calculate_concat_score(
        self, hidden_state_vector: torch.Tensor, encoder_output_vector: torch.Tensor
    ) -> torch.Tensor:
        """Reference: https://arxiv.org/pdf/1508.04025.pdf"""
        energy = self.attention(
            torch.cat(
                (
                    hidden_state_vector.expand(encoder_output_vector.size(0), -1, -1),
                    encoder_output_vector,
                ),
                2,
            )
        ).tanh()
        return torch.sum(self.parameter * energy, dim=2)

    def forward(
        self, hidden_state_vector: torch.Tensor, encoder_output_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate attention weights or energies
        and return the normalized probability distribution of the outputs (softmax).
        """
        if self.method == "general":
            attention_weights = self.calculate_general_score(
                hidden_state_vector, encoder_output_vector
            )
        elif self.method == "concat":
            attention_weights = self.calculate_concat_score(
                hidden_state_vector, encoder_output_vector
            )
        elif self.method == "dot":
            attention_weights = self.calculate_dot_score(
                hidden_state_vector, encoder_output_vector
            )

        attention_weights = attention_weights.t()

        return torch.nn.functional.softmax(attention_weights, dim=1).unsqueeze(1)
