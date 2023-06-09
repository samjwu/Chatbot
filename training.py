"""Utility functions for training an attention model."""

import os
import random

import torch

import processing
import vocabulary
from decoder import Decoder
from encoder import Encoder
from vocabulary import Vocabulary


def calculate_negative_log_likelihood_loss(
    input_vector: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """
    Calculate the average negative log likelihood
    of elements that map to 1s in the mask tensor.
    """
    total = mask.sum()
    crossEntropy = -torch.log(
        torch.gather(input_vector, 1, target.view(-1, 1)).squeeze(1)
    )
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, total.item()


def train(
    input_variable: torch.Tensor,
    input_lengths: torch.Tensor,
    target_variable: torch.Tensor,
    mask: torch.Tensor,
    max_target_len: int,
    encoder: Encoder,
    decoder: Decoder,
    embedding: torch.nn.Embedding,
    encoder_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    batch_size: int,
    clip_value: float,
    teacher_forcing_ratio: float,
    device: torch.device,
    max_length: int = 10,
) -> torch.Tensor:
    """
    Perform one training iteration.
    Use teacher forcing for some probability
    and gradient clipping for faster convergence.
    """
    # initial configurations
    # zero out gradients to avoid tracking unneeded information
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    input_lengths = input_lengths.to("cpu")
    # variables
    loss = 0
    print_losses = []
    totals = 0

    encoder_output_vector, encoder_hidden_state_vector = encoder(
        input_variable, input_lengths
    )

    # create decoder's initial input (begin each sentence with START token)
    decoder_input_vector = torch.LongTensor(
        [[vocabulary.START for _ in range(batch_size)]]
    )
    decoder_input_vector = decoder_input_vector.to(device)

    # set decoder's initial hidden state to encoder's final hidden state
    decoder_hidden_state_vector = encoder_hidden_state_vector[: decoder.num_layers]

    # if using teacher forcing this iteration, set decoder's next input as the current target
    # else set decoder's next input as decoder's current output
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # make decoder do a forward pass over a batch of sequences one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output_vector, decoder_hidden_state_vector = decoder(
                decoder_input_vector, decoder_hidden_state_vector, encoder_output_vector
            )

            # since teacher forcing is true, set next input to current target
            decoder_input_vector = target_variable[t].view(1, -1)

            # calculate and accumulate loss
            mask_loss, total = calculate_negative_log_likelihood_loss(
                decoder_output_vector, target_variable[t], mask[t], device
            )
            loss += mask_loss
            print_losses.append(mask_loss.item() * total)
            totals += total
    else:
        for t in range(max_target_len):
            decoder_output_vector, decoder_hidden_state_vector = decoder(
                decoder_input_vector, decoder_hidden_state_vector, encoder_output_vector
            )
            # since teacher forcing is false, set next input to current output
            _, topi = decoder_output_vector.topk(1)
            decoder_input_vector = torch.LongTensor(
                [[topi[i][0] for i in range(batch_size)]]
            )
            decoder_input_vector = decoder_input_vector.to(device)

            # calculate and accumulate loss
            mask_loss, total = calculate_negative_log_likelihood_loss(
                decoder_output_vector, target_variable[t], mask[t], device
            )
            loss += mask_loss
            print_losses.append(mask_loss.item() * total)
            totals += total

    # do backpropagation
    loss.backward()

    # clip gradients for the encoder and decoder (gradients are modified in place)
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=clip_value)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=clip_value)

    # update to adjust the model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / totals


def train_num_iterations(
    model_name: str,
    vocab: Vocabulary,
    questions_and_answers: list[list[str]],
    encoder: Encoder,
    decoder: Decoder,
    encoder_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    embedding: torch.nn.Embedding,
    num_encoder_layers: int,
    num_decoder_layers: int,
    save_directory: str,
    num_iterations: int,
    batch_size: int,
    print_iteration: int,
    save_iteration: int,
    clip_value: float,
    dataset_name: str,
    checkpoint: any,
    teacher_forcing_ratio: float,
    device: torch.device,
    hidden_size: int,
):
    """Run a given number of training iterations."""
    # load training batches for each iteration
    training_batches = [
        processing.convert_batch_to_training_data(
            vocab, [random.choice(questions_and_answers) for _ in range(batch_size)]
        )
        for _ in range(num_iterations)
    ]

    start_iteration = 1
    print_loss = 0
    if checkpoint is not None:
        start_iteration = checkpoint["iteration"] + 1

    for iteration in range(start_iteration, num_iterations + 1):
        training_batch = training_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # run one training iteration using the training batch
        loss = train(
            input_variable,
            lengths,
            target_variable,
            mask,
            max_target_len,
            encoder,
            decoder,
            embedding,
            encoder_optimizer,
            decoder_optimizer,
            batch_size,
            clip_value,
            teacher_forcing_ratio,
            device,
        )
        print_loss += loss

        if iteration % print_iteration == 0:
            print_loss_avg = print_loss / print_iteration
            print(
                "Iteration: {}; Completion: {:.1f}%; Average loss: {:.4f}".format(
                    iteration, iteration / num_iterations * 100, print_loss_avg
                )
            )
            print_loss = 0

        # save the checkpoint to run inference or train later
        if iteration % save_iteration == 0:
            directory = os.path.join(
                save_directory,
                model_name,
                dataset_name,
                "{}-{}_{}".format(num_encoder_layers, num_decoder_layers, hidden_size),
            )
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(
                {
                    "iteration": iteration,
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "encoder_optimizer": encoder_optimizer.state_dict(),
                    "decoder_optimizer": decoder_optimizer.state_dict(),
                    "loss": loss,
                    "vocabulary_dictionary": vocab.__dict__,
                    "embedding": embedding.state_dict(),
                },
                os.path.join(directory, "{}_{}.tar".format(iteration, "checkpoint")),
            )
