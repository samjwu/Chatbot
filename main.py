import codecs
import csv
import os
import random

import torch

import processing
import training
import vocabulary
from vocabulary import Vocabulary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = "movie-corpus"
processed_data_output = os.path.join(dataset, "formatted_movie_lines.txt")
movie_lines, movie_conversations = processing.extract_movie_lines_and_conversations(
    os.path.join(dataset, "utterances.jsonl")
)

delimiter = str(codecs.decode("\t", "unicode_escape"))
with open(processed_data_output, "w", encoding="utf-8") as output_file:
    writer = csv.writer(output_file, delimiter=delimiter, lineterminator="\n")
    for question_and_answer in processing.extract_questions_and_answers(
        movie_conversations
    ):
        writer.writerow(question_and_answer)

vocab, questions_and_answers = processing.process_data(processed_data_output, dataset)
print("Questions and Answers:")
for question_and_answer in questions_and_answers[:10]:
    print(question_and_answer)
print("\n")
questions_and_answers = processing.trim_words(vocab, questions_and_answers, 3)

small_batch_size = 5
(
    input_variable,
    lengths,
    output_variable,
    mask,
    max_target_len,
) = training.convert_batch_to_training_data(
    vocab, [random.choice(question_and_answer) for _ in range(small_batch_size)]
)
print("input_variable:", input_variable)
print("lengths:", lengths)
print("output_variable:", output_variable)
print("padding mask:", mask)
print("max_target_len:", max_target_len)
