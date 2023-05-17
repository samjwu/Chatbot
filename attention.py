"""
Attention is to enhance certain parts of the input sequence and diminish other parts.
Calculated using output vectors from the encoder and hidden state from the decoder.
"""

import torch
import torch.nn
import torch.nn.functional


class Attention(nn.Module):
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
        self, hidden_state_vectors: Tensor, encoder_output_vectors: Tensor
    ) -> Tensor:
        return torch.sum(hidden_state_vectors * encoder_output_vectors, dim=2)

    def calculate_general_score(
        self, hidden_state_vectors: Tensor, encoder_output_vectors: Tensor
    ) -> Tensor:
        energy = self.attention(encoder_output_vectors)
        return torch.sum(hidden_state_vectors * energy, dim=2)

    def calculate_concat_score(
        self, hidden_state_vectors: Tensor, encoder_output_vectors: Tensor
    ) -> Tensor:
        energy = self.attention(
            torch.cat(
                (
                    hidden_state_vectors.expand(encoder_output_vectors.size(0), -1, -1),
                    encoder_output_vectors,
                ),
                2,
            )
        ).tanh()
        return torch.sum(self.parameter * energy, dim=2)

    def forward_pass(
        self, hidden_state_vectors: Tensor, encoder_output_vectors: Tensor
    ) -> Tensor:
        """
        Calculate attention weights or energies
        and return the normalized probability distribution of the outputs (softmax).
        """
        if self.method == "general":
            attention_weights = self.general_score(
                hidden_state_vectors, encoder_output_vectors
            )
        elif self.method == "concat":
            attention_weights = self.concat_score(
                hidden_state_vectors, encoder_output_vectors
            )
        elif self.method == "dot":
            attention_weights = self.dot_score(
                hidden_state_vectors, encoder_output_vectors
            )

        attention_weights = attention_weights.t()

        return torch.nn.functional.softmax(attention_weights, dim=1).unsqueeze(1)
