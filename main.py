import codecs
import csv
import json
import os
import unicodedata

import torch

import vocabulary


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


def extract_q_and_a(movie_conversations: dict[str, str]) -> list[list[str]]:
    questions_and_answers = list()

    for conversation in movie_conversations.values():
        for i in range(len(conversation["lines"]) - 1):  # ignore the last line because it has no answer
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = "movie-corpus"
processed_data_output = os.path.join(dataset, "formatted_movie_lines.txt")

movie_lines, movie_conversations = extract_movie_lines_and_conversations(os.path.join(dataset, "utterances.jsonl"))

delimiter = str(codecs.decode("\t", "unicode_escape"))
with open(processed_data_output, "w", encoding="utf-8") as output_file:
    writer = csv.writer(output_file, delimiter=delimiter, lineterminator="\n")
    for pair in extract_q_and_a(movie_conversations):
        writer.writerow(pair)
