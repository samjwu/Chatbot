import os

import torch


def printLines(file: str, n: int) -> None:
    """Print the first n lines of a file"""
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

printLines(os.path.join("movie-corpus", "utterances.jsonl"), 3)
