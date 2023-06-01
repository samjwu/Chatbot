import codecs
import csv
import os
import random

import torch
import torch.nn

import processing
import training
import validating
import vocabulary
from decoder import Decoder
from encoder import Encoder
from greedy_search import GreedySearch
from vocabulary import Vocabulary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# processing data
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
) = processing.convert_batch_to_training_data(
    vocab, [random.choice(question_and_answer) for _ in range(small_batch_size)]
)
print("input_variable:", input_variable)
print("lengths:", lengths)
print("output_variable:", output_variable)
print("padding mask:", mask)
print("max_target_len:", max_target_len)

# model configurations
model_name = "chatbot_model"
attention_model = "dot"
hidden_size = 500
encoder_num_layers = 2
decoder_num_layers = 2
dropout = 0.1
batch_size = 64
checkpoint = None
checkpoint_name = None
checkpoint_iterations = 40
save_directory = os.path.join("data", "save")

checkpoint_name = os.path.join(
    save_directory,
    model_name,
    dataset,
    "{}-{}_{}".format(encoder_num_layers, decoder_num_layers, hidden_size),
    "{}_checkpoint.tar".format(checkpoint_iterations),
)

has_checkpoint = os.path.isfile(checkpoint_name)

# load model
if has_checkpoint:
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_name, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(checkpoint_name)
    encoder_state_dict = checkpoint["encoder"]
    decoder_state_dict = checkpoint["decoder"]
    encoder_optimizer_state_dict = checkpoint["encoder_optimizer"]
    decoder_optimizer_state_dict = checkpoint["decoder_optimizer"]
    vocab.__dict__ = checkpoint["vocabulary_dictionary"]
    embedding_state_dict = checkpoint["embedding"]

# build encoder and decoder
embedding = torch.nn.Embedding(vocab.num_words, hidden_size)
if has_checkpoint:
    embedding.load_state_dict(embedding_state_dict)
encoder = Encoder(hidden_size, embedding, encoder_num_layers, dropout)
decoder = Decoder(
    attention_model,
    embedding,
    hidden_size,
    vocab.num_words,
    decoder_num_layers,
    dropout,
)
if has_checkpoint:
    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)
encoder = encoder.to(device)
decoder = decoder.to(device)

# training configurations
learning_rate = 0.0001
decoder_learning_ratio = 5.0
num_iterations = 40
print_iteration = 1
save_iteration = 500
clip_value = 50.0
teacher_forcing_ratio = 1.0

# set dropout layers in training mode
encoder.train()
decoder.train()

# initialize optimizers
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.Adam(
    decoder.parameters(), lr=learning_rate * decoder_learning_ratio
)
if has_checkpoint:
    encoder_optimizer.load_state_dict(encoder_optimizer_state_dict)
    decoder_optimizer.load_state_dict(decoder_optimizer_state_dict)

if torch.cuda.is_available():
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

# train the model
training.train_num_iterations(
    model_name,
    vocab,
    questions_and_answers,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    embedding,
    encoder_num_layers,
    decoder_num_layers,
    save_directory,
    num_iterations,
    batch_size,
    print_iteration,
    save_iteration,
    clip_value,
    dataset,
    checkpoint,
    teacher_forcing_ratio,
    device,
    hidden_size,
)

# set dropout layers in validation mode
encoder.eval()
decoder.eval()

searcher = GreedySearch(encoder, decoder, device)

# run chatbot
validating.validate_input(encoder, decoder, searcher, vocab, device)
