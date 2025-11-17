import random

class HonestAgent:
    def __init__(self, name, vocab, text_len, lm):
        self.name = name
        self.vocab = vocab
        self.text_len = text_len
        self.lm = lm
    
    def propose(self):
        crib = random.choice(self.vocab)
        side = random.choice([1,2]) #agent picks one of the two ciphertexts
        offset = random.randint(0, self.text_len - len(crib))
        return side, offset, crib
    
    def vote(self, fragment, lm_threshold):
        """Vote yes if the fragment looks like English"""
        score = self.lm.score(fragment)
        return score >= lm_threshold
