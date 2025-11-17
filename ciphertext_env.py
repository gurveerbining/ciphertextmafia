import os, random, string
import nltk
from nltk.corpus import brown
# make sure corpus is downloaded once
try:
    nltk.data.find("corpora/brown")
except LookupError:
    nltk.download("brown")

class CipherEnv:
    def __init__(self, text_len=80, max_rounds=10):
        self.text_len = text_len
        self.max_rounds = max_rounds
        self.alphabet = string.ascii_uppercase + " "


    def _random_sentence(self):
        """Generate an English-like plaintext using Brown corpus words."""
        # choose random start index in the corpus
        words = [w.upper() for w in brown.words() if w.isalpha()]
        idx = random.randint(0, len(words) - 30)
        s = " ".join(words[idx : idx + 30])  # 30 words is plenty
        s = s.replace("\n", " ")
        # trim/pad to fixed length
        s = s[: self.text_len]
        if len(s) < self.text_len:
            s = s.ljust(self.text_len)
        return s
    
    def reset(self):
        # 1) make two plaintexts
        self.p1 = self._random_sentence()
        self.p2 = self._random_sentence()

        # 2) make shared one-time-pad key
        self.key = os.urandom(self.text_len)

        # 3) encrypt both with same key
        self.c1 = bytes([ord(ch) ^ k for ch, k in zip(self.p1, self.key)])
        self.c2 = bytes([ord(ch) ^ k for ch, k in zip(self.p2, self.key)])

        # 4) what agents actually see
        self.x = bytes([a ^ b for a, b in zip(self.c1, self.c2)])

        # 5) initialize masks
        self.mask1 = ["_"] * self.text_len
        self.mask2 = ["_"] * self.text_len
        self.round = 0

        print(f"p1: {self.p1}")
        print(f"\np2: {self.p2}")

        return self.x
    
    def implied_fragment(self, side, offset, crib):
        frag=""
        for i, ch in enumerate(crib):
            index = offset + i
            if index >= self.text_len:
                break
            if side == 1:
                frag += chr(self.x[index] ^ ord(ch))
            else:
                frag += chr(self.x[index] ^ ord(ch))
        
        return frag


