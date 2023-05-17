"""
Utility functions for training an attention model.
"""

import itertools

import torch

import vocabulary
from vocabulary import Vocabulary


def sentence_to_indices(vocab: Vocabulary, sentence: str) -> list[list[int]]:
    return [vocab.word_to_index[word] for word in sentence.split(" ")] + [
        vocabulary.END
    ]


def add_padding(tensor: list[list[int]], fillvalue: int) -> list[int]:
    return list(itertools.zip_longest(*tensor, fillvalue=fillvalue))


def construct_binary_matrix(tensor: list[list[int]]) -> list[list[int]]:
    matrix = []

    for i, seq in enumerate(tensor):
        matrix.append([])

        for token in seq:
            if token == vocabulary.PAD:
                matrix[i].append(0)
            else:
                matrix[i].append(1)

    return matrix


def generate_input_tensor(
    sentences: list[str], vocab: Vocabulary
) -> tuple[list[int], list[int]]:
    """
    Convert an input list into a padded sequence tensor.
    Return the tensor and the lengths of each batch.
    """
    batches = [sentence_to_indices(vocab, sentence) for sentence in sentences]
    lengths = torch.tensor([len(indices) for indices in batches])
    padded_list = add_padding(batches, 0)
    padded_input = torch.LongTensor(padded_list)
    return padded_input, lengths


def generate_output_tensor(
    sentences: list[str], vocab: Vocabulary
) -> tuple[list[int], list[int], int]:
    """
    Convert an output list into a padded sequence tensor.
    Return the tensor, a padding mask, and the max target length.
    """
    batches = [sentence_to_indices(vocab, sentence) for sentence in sentences]
    max_target_len = max([len(indices) for indices in batches])
    padded_list = add_padding(batches, 0)
    mask = construct_binary_matrix(padded_list)
    mask = torch.BoolTensor(mask)
    padded_output = torch.LongTensor(padded_list)
    return padded_output, mask, max_target_len


def convert_batch_to_training_data(
    vocab: Vocabulary, batch: list[list[str]]
) -> tuple[list[int], list[int], list[int], list[int], int]:
    batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)

    question_batch = list()  # input
    answer_batch = list()  # output

    for question_and_answer in batch:
        question_batch.append(question_and_answer[0])
        answer_batch.append(question_and_answer[1])

    input_data, lengths = generate_input_tensor(question_batch, vocab)
    output_data, mask, max_target_len = generate_output_tensor(answer_batch, vocab)

    return input_data, lengths, output_data, mask, max_target_len
