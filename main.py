import codecs
import csv
import json
import itertools
import os
import re
import unicodedata

import torch

from vocabulary import Vocabulary


def extract_movie_lines_and_conversations(file_name: str) -> tuple[dict[str, str], dict[str, str]]:
    movie_lines = dict()
    movie_conversations = dict()

    with open(file_name, "r", encoding="iso-8859-1") as input_file:
        for line in input_file:
            line_json = json.loads(line)
            
            line_object = dict()
            line_object["lineID"] = line_json["id"]
            line_object["characterID"] = line_json["speaker"]
            line_object["text"] = line_json["text"]
            movie_lines[line_object["lineID"]] = line_object

            if line_json["conversation_id"] not in movie_conversations:
                conversation_object = dict()
                conversation_object["conversationID"] = line_json["conversation_id"]
                conversation_object["movieID"] = line_json["meta"]["movie_id"]
                conversation_object["lines"] = [line_object]
            else: # movie line is continuing a conversation
                conversation_object = movie_conversations[line_json["conversation_id"]]
                conversation_object["lines"].insert(0, line_object)
            movie_conversations[conversation_object["conversationID"]] = conversation_object

    return movie_lines, movie_conversations


def extract_questions_and_answers(movie_conversations: dict[str, str]) -> list[list[str]]:
    questions_and_answers = list()

    for conversation in movie_conversations.values():
        # ignore the last line because it has no answer
        for i in range(len(conversation["lines"]) - 1):
            question = conversation["lines"][i]["text"].strip()
            answer = conversation["lines"][i+1]["text"].strip()
            
            if question is not None and answer is not None:
                questions_and_answers.append([question, answer])

    return questions_and_answers


def convert_unicode_to_ascii(s: str) -> None:
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_str(s: str) -> str:
    s = convert_unicode_to_ascii(s.lower().strip())
    # prepend punctuation with space
    s = re.sub(r"([.!?])", r" \1", s)
    # remove chars that are not letters or punctuation
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    # condense multiple spaces into one space
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def generate_vocabulary(data_file: str, dataset_name: str) -> tuple[Vocabulary, list[list[str]]]:
    """
    Splits each line in file by tabs and then normalize them.
    Then return the questions and answers with a new Vocabulary.
    """
    lines = open(data_file, encoding='utf-8').read().strip().split('\n')
    questions_and_answers = [[normalize_str(s) for s in l.split('\t')] for l in lines]
    vocab = Vocabulary(dataset_name)
    return vocab, questions_and_answers


def is_short(question_and_answer: list[str], threshold: int) -> bool:
    """Return true if both question and answer is shorter than threshold."""
    question = question_and_answer[0]
    answer = question_and_answer[1]
    return len(question.split(' ')) < threshold and len(answer.split(' ')) < threshold


def filter_questions_and_answers(questions_and_answers: list[list[str]]) -> list[list[str]]:
    """Keep only questions and answers longer than a min threshold."""
    return [question_and_answer for question_and_answer in questions_and_answers if is_short(question_and_answer, 10)]


def process_data(data_file: str, dataset_name: str) -> tuple[Vocabulary, list[list[str]]]:
    vocab, questions_and_answers = generate_vocabulary(data_file, dataset_name)
    print(f"{len(questions_and_answers)} questions and answers read from data file")
    
    questions_and_answers = filter_questions_and_answers(questions_and_answers)
    print(f"{len(questions_and_answers)} questions and answers remaining after filtering")
    
    for question_and_answer in questions_and_answers:
        vocab.add_sentence(question_and_answer[0])
        vocab.add_sentence(question_and_answer[1])
    print(f"{vocab.num_words} words in total")

    return vocab, questions_and_answers


def trim_words(vocab: Vocabulary, questions_and_answers: list[list[str]], threshold: int) -> list[list[str]]:
    """
    Get rid of questions and answers 
    with words that have frequency that fall below a given threshold.
    """
    vocab.trim(threshold)
    
    keep_questions_and_answers = []
    for question_and_answer in questions_and_answers:
        question = question_and_answer[0]
        answer = question_and_answer[1]
        
        keep_question = True
        keep_answer = True
        
        for word in question.split(' '):
            if word not in vocab.word_to_index:
                keep_question = False
                break
        
        for word in answer.split(' '):
            if word not in vocab.word_to_index:
                keep_answer = False
                break

        if keep_question and keep_answer:
            keep_questions_and_answers.append(question_and_answer)

    total_keep = len(keep_questions_and_answers)
    total_sentences = len(questions_and_answers)
    print(f"Kept {total_keep} out of {total_sentences} questions and answers = {(total_keep / total_sentences * 100):.4f}%")
    return keep_questions_and_answers


def sentence_to_indices(vocab: Vocabulary, sentence: str) -> list[int]:
    return [vocab.word_to_index[word] for word in sentence.split(' ')] + [vocabulary.END]


def add_padding(tensor: list[int], fillvalue: int) -> list[int]:
    return list(itertools.zip_longest(*tensor, fillvalue))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = "movie-corpus"
processed_data_output = os.path.join(dataset, "formatted_movie_lines.txt")

movie_lines, movie_conversations = extract_movie_lines_and_conversations(os.path.join(dataset, "utterances.jsonl"))

delimiter = str(codecs.decode("\t", "unicode_escape"))
with open(processed_data_output, "w", encoding="utf-8") as output_file:
    writer = csv.writer(output_file, delimiter=delimiter, lineterminator="\n")
    for question_and_answer in extract_questions_and_answers(movie_conversations):
        writer.writerow(question_and_answer)

vocab, questions_and_answers = process_data(processed_data_output, dataset)
print("Questions and Answers:")
for question_and_answer in questions_and_answers[:10]:
    print(question_and_answer)

questions_and_answers = trim_words(vocab, questions_and_answers, 3)
