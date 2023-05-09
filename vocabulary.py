"""
Keep track of words in the dataset
and commonly used words.
"""

PAD = 0
START = 1
END = 2

class Vocabulary:
    def __init__(self, name: str) -> None:
        self.name = name
        self.trimmed = False
        self.initialize()
    
    def initialize(self) -> None:
        self.word_to_index = dict()
        self.index_to_word = {PAD: "PAD", START: "START", END: "END"}
        self.word_count = dict()
        self.num_words = 3

    def add_word(self, word: str) -> None:
        if word not in self.word_to_index.keys():
            self.word_to_index[word] = self.num_words
            self.index_to_word[self.num_words] = word
            self.word_count[word] = 1
            self.num_words += 1
        else:
            self.word_count[word] += 1

    def add_sentence(self, sentence: str) -> None:
        for word in sentence.split(' '):
            self.add_word(word)

    def trim(self, min_count: int) -> None:
        """Get rid of words with frequency below a given minimum count."""
        if self.trimmed:
            return

        self.trimmed = True

        keep_words = []
        for word, count in self.word_count.items():
            if count >= min_count:
                keep_words.append(word)

        total_keep = len(keep_words)
        total_words = len(self.word_to_index)
        print(f"keep {total_keep} out of {total_words} words = {(total_keep / total_words):.4g}%")

        self.initialize()

        for word in keep_words:
            self.add_word(word)
