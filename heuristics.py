import string

COMMON_BIGRAMS = {
    "TH", "HE", "IN", "ER", "AN", "RE", "ON", "AT", "ND", "OR",
    "HA", "EN", "ES", "ST", "TO", "NT", "ED", "IS", "IT", "AL"
}

COMMON_TRIGRAMS = {
    "THE", "AND", "ING", "ION", "ENT", "HER", "FOR", "THA", "EST", "INT"
}

class HeuristicScorer:
    def __init__(self, env):
        self.env = env
    
    def score(self, side, offset, crib):
        """Returns heuristic score of proposing crib at a position"""
        mask = self.env.mask1 if side ==1 else self.env.mask2
        length = len(crib)
        score = 0.0

        #1) Mask consistency check
        mismatch = False
        for i in range(length):
            if offset + i >= len(mask):
                mismatch = True
                score -= 3.0
                continue
            m = mask[offset + i]
            if m != "_" and m != crib[i]:
                mismatch = True
                score -= 5.0
            elif m == crib[i]:
                score += 3.0
        
        #2) Character validity check
        for ch in crib:
            if "A" <= ch <= "Z":
                score += 1.0
            
            elif ch == " ":
                score += 0.5
            
            else:
                score -= 2.0
        
        #3) Check bigrams and trigrams
        up = crib.upper()

        for i in range(len(up) - 1):
            if up[i:i+2] in COMMON_BIGRAMS:
                score += 1.0
        
        for i in range(len(up) - 2):
            if up[i:i+3] in COMMON_TRIGRAMS:
                score += 2.0
        
        #4) Space plausibility, avoid two spaces in a row
        if "  " in crib:
            score -= 4.0
        
        #5) Reward for cribs that overlap as many unknown positions as possible
        reveals = 0
        for i in range(length):
            if mask[offset+i] == "_" and crib[i] in string.ascii_uppercase:
                reveals += 1
        score += reveals * 4.0

        #5) Normalize
        return score / max(1, len(crib))