import json
import os
from typing import List

class DevanagariTokenizer:
    """
    Tokenizer for Devanagari script (and other characters present in the labels).
    Index 0 is strictly reserved for the CTC <blank> token.
    Index 1 is reserved for <unk>.
    """
    def __init__(self):
        self.char_to_idx = {"<blank>": 0, "<unk>": 1}
        self.idx_to_char = {0: "<blank>", 1: "<unk>"}
        self.vocab_size = 2

    def build_vocab(self, texts: List[str]):
        """
        Builds the character vocabulary from a list of strings.
        """
        for text in texts:
            for char in text:
                if char not in self.char_to_idx:
                    self.char_to_idx[char] = self.vocab_size
                    self.idx_to_char[self.vocab_size] = char
                    self.vocab_size += 1

    def encode(self, text: str) -> List[int]:
        """
        Converts a text string to a list of integer tokens.
        """
        return [self.char_to_idx.get(c, 1) for c in text]

    def decode(self, indices: List[int], remove_repeats: bool = True) -> str:
        """
        Converts a list of integer tokens back to text.
        If remove_repeats is True, performs greedy CTC decoding logic 
        (removes sequential repeats and blanks).
        """
        if not remove_repeats:
            return "".join([self.idx_to_char.get(i, "") for i in indices if i != 0])
            
        decoded = []
        prev = -1
        for idx in indices:
            if idx != prev and idx != 0:
                decoded.append(idx)
            prev = idx
        return "".join([self.idx_to_char.get(i, "") for i in decoded])

    def save_vocab(self, path: str):
        """
        Saves the tokenizer vocabulary mapping to a JSON file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "char_to_idx": self.char_to_idx,
                "idx_to_char": {str(k): v for k, v in self.idx_to_char.items()},
                "vocab_size": self.vocab_size
            }, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: str):
        """
        Loads the tokenizer vocabulary mapping from a JSON file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocab file not found at {path}")
            
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.char_to_idx = data["char_to_idx"]
            self.idx_to_char = {int(k): v for k, v in data["idx_to_char"].items()}
            self.vocab_size = data["vocab_size"]
