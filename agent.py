import random
from heuristics import HeuristicScorer

class HonestAgent:
    def __init__(self, name, vocab, text_len, env, epsilon = 0.2, alpha = 0.3, gamma = 0.0):
        self.name = name
        self.vocab = vocab
        self.text_len = text_len
        self.env = env
        self.heuristic = HeuristicScorer(env)

        #Q-Learning parameters
        self.epsilon = epsilon #exploration rate
        self.alpha = alpha #learning rate
        self.gamma = gamma #discount

        self.Q = {crib:0.0 for crib in vocab}
        self.last_crib = None #remember last action for the update
        self.trust = {}
        self.current_mask = ["_"] * text_len #updated every step

    def propose(self):
        """epsilon-greedy crib selection using Q-values"""
        #1. choose crib
        if random.random() < self.epsilon:
            crib = random.choice(self.vocab)
        else:
            crib = max(self.vocab, key = lambda c: self.Q.get(c, 0.0))
        
        self.last_crib = crib
        clen = len(crib)

        #2. find offsets where at least 1 char is unknown
        unknown_offsets = []
        for i in range(self.text_len - clen):
            if "_" in self.current_mask[i: i + clen]:
                unknown_offsets.append(i)
        # if none found, random offset
        if unknown_offsets:
            offset = random.choice(unknown_offsets)
        else:
            offset = random.randint(0, self.text_len - clen)
        
        side = random.choice([1,2])
        
        
        return side, offset, crib   

    def update_q(self, action, reward):
        """Q(a) ← Q(a) + alpha[reward - Q(a)]"""
        if self.last_crib is None:
            return

        #normalize reward by crib length
        length = len(self.last_crib)
        scaled_reward = reward / max(1, length)

        old_q = self.Q.get(self.last_crib, 0.0)
        new_q = old_q + self.alpha * (scaled_reward - old_q)
        self.Q[self.last_crib] = new_q
    
    def vote(self, side, offset, crib):
        """Vote yes/no based on heuristic score of crib"""
        score = self.heuristic.score(side, offset, crib)
        return score >= -1.5

    def update_trust(self, proposer_name, reward):
        """increase trust when proposer's accepted proposal gives a positive reward, decrease for negative reward"""
        if proposer_name == self.name:
            return
            #don't track self
        
        current = self.trust.get(proposer_name, 0.0)
        delta = max(-2, min(2, reward))
        self.trust[proposer_name] = current + 0.1 * delta
    
    def most_suspicious(self):
        """return name of least trusted agent"""
        if not self.trust:
            return None
            #just in case there is no trust info for some reason

        return min(self.trust, key = lambda n: self.trust[n])

class ImposterAgent:
    """
    Imposter agent
    - randomly proposes crib
    - sometimes votes incorrectly to sabotage
    """
    def __init__(self, name, vocab, text_len, env, lie_vote_prob = 0.3):
        self.name = name
        self.vocab = vocab
        self.text_len = text_len
        self.lie_vote_prob = lie_vote_prob

        self.env = env
        self.heuristic = HeuristicScorer(env)

    def propose(self):
        """this is random for now, could maybe change this later"""
        crib = random.choice(self.vocab)
        side = random.choice([1,2]) #agent picks one of the two ciphertexts
        offset = random.randint(0, self.text_len - len(crib))
        return side, offset, crib
    
    def vote(self, side, offset, crib):
        """"Imposter lies sometimes and votes incorrectly"""
        honest_vote = self.heuristic.score(side, offset, crib) >= -1.5
        if random.random() < self.lie_vote_prob:
            return not honest_vote
        
        return honest_vote
    
    
