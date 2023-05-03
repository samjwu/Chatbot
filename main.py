import codecs
import json 
import os

import torch


def extractMovieLinesAndConversations(file_name: str) -> tuple[dict[str, str], dict[str, str]]:
    movie_lines = dict()
    movie_conversations = dict()

    with open(file_name, "r", encoding="iso-8859-1") as f:
        for line in f:
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = "movie-corpus"
processed_data_output = os.path.join(dataset, "formatted_movie_lines.txt")

delimiter = '\t'
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

movie_lines, movie_conversations = extractMovieLinesAndConversations(os.path.join(dataset, "utterances.jsonl"))
