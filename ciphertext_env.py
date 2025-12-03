import os, random, string
import nltk
from nltk.corpus import brown
# make sure corpus is downloaded once
try:
    nltk.data.find("corpora/brown")
except LookupError:
    nltk.download("brown")

class CipherEnv:
    def __init__(self, text_len=40, max_rounds=100):
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
        """Return the implied plaintext fragment by XORing x with crib"""
        frag=""
        for i, ch in enumerate(crib):
            index = offset + i
            if index >= self.text_len:
                break
            frag += chr(self.x[index] ^ ord(ch))
        
        return frag

    def apply_proposal(self, side, offset, crib):
        """Apply crib, Reward: +1 for correct character, -1 for incorrect character"""
        reward = 0 

        for i, ch in enumerate(crib):
            index = offset + i
            if index >= self.text_len:
                break
            if side == 1:
                true_char = self.p1[index]
                if ch == true_char:
                    #reveal position
                    if self.mask1[index] == "_":
                        self.mask1[index] = true_char
                        if self.mask2[index] == "_":
                            self.mask2[index] = self.p2[index]
                    reward += 1
                else:
                    reward -= 1
            else:
            #side == 2
                true_char = self.p2[index]
                if ch == true_char:
                    if self.mask2[index] == "_":
                        self.mask2[index] = true_char
                        if self.mask1[index] == "_":
                            self.mask1[index] = self.p1[index]
                    reward += 1
                else:
                    reward -= 1

        done = self.completion_ratio() >= 1.0
        return reward, done

    def completion_ratio(self):
        """fraction of characters correctly recovered in both plaintexts"""
        correct = 0
        total = 2 * self.text_len
        for i in range(self.text_len):
            if self.mask1[i] == self.p1[i]:
                correct += 1
            if self.mask2[i] == self.p2[i]:
                correct += 1
        
        return correct/total
    
    def print_masks(self):
        print("Mask1: ", " ".join(self.mask1))
        print("Mask2: ", " ".join(self.mask2))
        print(f"Completion: {self.completion_ratio() * 100}%")
    
        