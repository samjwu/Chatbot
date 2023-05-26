"""Utility functions for validating an attention model."""

import torch

from decoder import Decoder
from encoder import Encoder
from greedy_search import GreedySearch
from vocabulary import Vocabulary


def validate_sentence(
    encoder: Encoder,
    decoder: Decoder,
    searcher: GreedySearch,
    vocab: Vocabulary,
    sentence: str,
    max_length: int = 10,
) -> list[str]:
    """Evaluate an input string/sentence and return a list/batch of decoded words."""
    # convert the sentence into its indices and get their lengths
    indices_batch = [sentence_to_indices(vocab, sentence)]
    lengths = torch.tensor([len(indices) for indices in indices_batch])

    # transpose dimensions of input batch to match the attention model's expectations
    input_batch = torch.LongTensor(indices_batch).transpose(0, 1)

    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")

    # greedily decode the sentence and return it as a batch
    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [vocab.index_to_word[token.item()] for token in tokens]
    return decoded_words
