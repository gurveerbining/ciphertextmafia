import math
import nltk
from nltk.corpus import brown
from collections import defaultdict, Counter
# make sure corpus is downloaded once
try:
    nltk.data.find("corpora/brown")
except LookupError:
    nltk.download("brown")

class NgramLM:
    def __init__(self):
        self.bigrams = Counter()
        self.trigrams = Counter()
        
        #collect words from corpus
        words = [w.upper() for w in brown.words() if w.isalpha()]
        text = " ".join(words)

        for i in range(len(text)-2):
            bg = text[i:i+2]
            tg = text[i:i+3]
            self.bigrams[bg] +=1
            self.trigrams[tg] +=1
        
        #constant
        self.k = 1
    
    def score(self, text):
        """return LM score for a text fragment"""
        score = 0.0
        for i in range(len(text)-2):
            bg = text[i:i+2]
            tg = text[i:i+3]
            count_bg = self.bigrams[bg] + self.k
            count_tg = self.trigrams[tg] + self.k
            score += math.log(count_tg/count_bg)

        return score